use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::audio::{self, AudioEvent, CaptureHandle, PRECHECK_SECONDS, SAMPLE_RATE,
    SILENCE_THRESHOLD_DB, SILENCE_WARNING_SECONDS};
use crate::playback::{self as pb, PlaybackEvent, PlaybackHandle};
use crate::device::{self, MicDevice};
use crate::session::{self, Session};
use crate::wav::WavWriter;

// ── State machine ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppMode {
    /// First-run or explicit edit: user is entering book/chapter.
    Setup,
    /// Mic found, session set.  Waiting for user to press Space.
    Ready,
    /// Running the 2-second pre-recording signal check.
    PreCheck,
    /// Actively capturing audio to disk.
    Recording,
    /// Just stopped — prompting "continue chapter? (Y/n)"
    PostRecording,
    /// Playing back the last N seconds of the just-recorded clip so the user
    /// can find the punch-in point.
    PunchRollback,
    /// Approved mic is missing or was unplugged.
    MicError,
    /// Unrecoverable error (ALSA failure, disk full, etc.).
    Fatal,
}

/// Which text field is focused in Setup mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetupField {
    Book,
    Chapter,
}

// ── App ───────────────────────────────────────────────────────────────────────

pub struct App {
    // ── Session ───────────────────────────────────────────────────────────
    pub session: Session,

    // ── Mode ──────────────────────────────────────────────────────────────
    pub mode: AppMode,

    // ── Device ────────────────────────────────────────────────────────────
    pub mic: Option<MicDevice>,

    // ── Setup form ────────────────────────────────────────────────────────
    pub setup_book: String,
    pub setup_chapter: String,
    pub setup_field: SetupField,
    pub setup_error: Option<String>,

    // ── Live capture resources (Some when in PreCheck / Recording) ────────
    pub capture: Option<CaptureHandle>,
    pub wav: Option<WavWriter>,
    pub active_file: Option<PathBuf>,

    // ── Signal / activity state ───────────────────────────────────────────
    pub last_rms_db: f32,
    /// Spinner tick — incremented each time a non-silent batch arrives.
    pub activity_tick: u8,
    /// True when the runtime silence watchdog has fired.
    pub silence_warning: bool,
    pub last_non_silence: Instant,

    // ── Pre-check state ───────────────────────────────────────────────────
    pub precheck_frames_seen: usize,
    pub precheck_signal_found: bool,
    pub precheck_target_frames: usize,

    // ── Recording timing ──────────────────────────────────────────────────
    pub record_start: Option<Instant>,

    // ── Phase 2: punch-and-roll ───────────────────────────────────────────
    /// How far back (seconds) to rewind before rollback playback.
    /// Loaded from PUNCH_BACK_TIME env var (value in seconds, e.g. "15s"); defaults to 15.0.
    pub punch_back_time: f64,
    /// Crossfade duration in milliseconds applied at punch-in/out boundaries.
    /// Loaded from CROSSFADE_TIME env var (value in ms, e.g. "10ms"); defaults to 10.0.
    pub crossfade_time: f64,
    /// Absolute timeline position (seconds) when the current/last clip started.
    pub clip_start_timeline: f64,
    /// Duration (seconds) of the clip that triggered a punch operation.
    pub last_clip_duration: f64,
    /// Absolute timeline position where rollback playback began.
    pub punch_rollback_abs: f64,
    /// Active playback thread handle (Some during PunchRollback).
    pub playback: Option<PlaybackHandle>,
    /// When playback started (used to compute elapsed and punch-in time).
    pub playback_start: Option<Instant>,
    /// True once the playback thread has signalled PlaybackEvent::Done.
    pub playback_done: bool,
    /// Elapsed seconds at the moment PlaybackEvent::Done was received.
    pub playback_elapsed: f64,
    // ── Status / error messages ───────────────────────────────────────────
    pub status_msg: Option<String>,
    pub error_msg: String,

    // ── One-time pre-check ────────────────────────────────────────────────
    /// True after the first pre-check has completed.  Subsequent recordings
    /// skip the signal check and start capturing immediately.
    pub precheck_done: bool,

    // ── Short-clip discard ────────────────────────────────────────────────
    /// Loaded from DISCARD_SHORT_CLIPS env var; defaults to false.
    pub discard_short_clips: bool,
    /// Minimum clip duration (seconds) to keep. Loaded from DISCARD_DURATION
    /// env var (value in seconds, e.g. "1s"); defaults to 1.0.
    pub discard_duration_secs: f64,

    // ── Quit flag ─────────────────────────────────────────────────────────
    pub should_quit: bool,
}

impl App {
    pub fn new() -> Self {
        let session = session::load().unwrap_or_else(|| Session::new(String::new(), 1));
        let mode = if session.book.is_empty() {
            AppMode::Setup
        } else {
            AppMode::Ready
        };

        // Pre-populate setup form with saved values.
        let setup_book = session.book.clone();
        let setup_chapter = format!("{:02}", session.chapter);

        // Read PUNCH_BACK_TIME from environment (already loaded from .env by main).
        // Value is in seconds with an optional "s" suffix (e.g. "15s" or "15").
        let punch_back_time = std::env::var("PUNCH_BACK_TIME")
            .ok()
            .and_then(|v| v.trim().trim_end_matches('s').parse::<f64>().ok())
            .unwrap_or(15.0)
            .max(1.0); // enforce minimum of 1 second

        // Read CROSSFADE_TIME from environment.
        // Value is in milliseconds with an optional "ms" suffix (e.g. "10ms" or "10").
        let crossfade_time = std::env::var("CROSSFADE_TIME")
            .ok()
            .and_then(|v| v.trim().trim_end_matches("ms").parse::<f64>().ok())
            .unwrap_or(10.0)
            .max(0.0);

        // Read DISCARD_SHORT_CLIPS / DISCARD_DURATION from environment.
        let discard_short_clips = std::env::var("DISCARD_SHORT_CLIPS")
            .map(|v| v.trim().eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let discard_duration_secs = std::env::var("DISCARD_DURATION")
            .ok()
            .and_then(|v| v.trim().trim_end_matches('s').parse::<f64>().ok())
            .unwrap_or(1.0)
            .max(0.0);

        App {
            session,
            mode,
            mic: None,
            setup_book,
            setup_chapter,
            setup_field: SetupField::Book,
            setup_error: None,
            capture: None,
            wav: None,
            active_file: None,
            last_rms_db: -100.0,
            activity_tick: 0,
            silence_warning: false,
            last_non_silence: Instant::now(),
            precheck_frames_seen: 0,
            precheck_signal_found: false,
            precheck_target_frames: (PRECHECK_SECONDS * SAMPLE_RATE as f64) as usize,
            record_start: None,
            punch_back_time,
            crossfade_time,
            clip_start_timeline: 0.0,
            last_clip_duration: 0.0,
            punch_rollback_abs: 0.0,
            playback: None,
            playback_start: None,
            playback_done: false,
            playback_elapsed: 0.0,
            status_msg: None,
            error_msg: String::new(),
            precheck_done: false,
            discard_short_clips,
            discard_duration_secs,
            should_quit: false,
        }
    }

    // ── Device detection ──────────────────────────────────────────────────────

    /// Attempt to detect the approved USB microphone.  Call on startup and
    /// whenever re-entering Ready mode.
    pub fn detect_mic(&mut self) {
        match device::find_approved_mic() {
            Ok(mic) => {
                self.mic = Some(mic);
                if self.mode == AppMode::MicError {
                    self.mode = AppMode::Ready;
                }
            }
            Err(e) => {
                self.mic = None;
                self.error_msg = e.to_string();
                self.mode = AppMode::MicError;
            }
        }
    }

    // ── Keyboard handling ─────────────────────────────────────────────────────

    /// Handle a raw key event.  Returns true if the app should quit.
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        // Ctrl-C always quits (raw mode suppresses SIGINT delivery).
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && key.code == KeyCode::Char('c')
        {
            self.emergency_stop();
            return true;
        }

        match self.mode {
            AppMode::Setup => self.handle_key_setup(key),
            AppMode::Ready => self.handle_key_ready(key),
            AppMode::PreCheck => self.handle_key_precheck(key),
            AppMode::Recording => self.handle_key_recording(key),
            AppMode::PostRecording => self.handle_key_post(key),
            AppMode::PunchRollback => self.handle_key_punch_rollback(key),
            AppMode::MicError => self.handle_key_mic_error(key),
            AppMode::Fatal => {
                if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) {
                    return true;
                }
            }
        }

        self.should_quit
    }

    fn handle_key_setup(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Tab => {
                self.setup_field = match self.setup_field {
                    SetupField::Book => SetupField::Chapter,
                    SetupField::Chapter => SetupField::Book,
                };
            }
            KeyCode::Enter => self.confirm_setup(),
            KeyCode::Esc => {
                // Revert to saved session and go to Ready (if session is valid).
                if !self.session.book.is_empty() {
                    self.setup_error = None;
                    self.mode = AppMode::Ready;
                }
            }
            KeyCode::Backspace => match self.setup_field {
                SetupField::Book => {
                    self.setup_book.pop();
                }
                SetupField::Chapter => {
                    self.setup_chapter.pop();
                }
            },
            KeyCode::Char(c) => match self.setup_field {
                SetupField::Book => {
                    // Replace spaces with underscores (used as directory name).
                    let ch = if c == ' ' { '_' } else { c };
                    if self.setup_book.len() < 64 && (ch.is_alphanumeric() || "_-".contains(ch)) {
                        self.setup_book.push(ch);
                    }
                }
                SetupField::Chapter => {
                    if c.is_ascii_digit() && self.setup_chapter.len() < 3 {
                        self.setup_chapter.push(c);
                    }
                }
            },
            _ => {}
        }
    }

    fn confirm_setup(&mut self) {
        let book = self.setup_book.trim().to_string();
        if book.is_empty() {
            self.setup_error = Some("Book name cannot be empty.".into());
            return;
        }

        let chapter: u32 = match self.setup_chapter.trim().parse() {
            Ok(n) if n >= 1 => n,
            _ => {
                self.setup_error = Some("Chapter must be a number >= 1.".into());
                return;
            }
        };

        self.session.book = book;
        self.session.chapter = chapter;
        // Keep the existing part counter only if the book/chapter match the
        // saved session exactly; otherwise reset to 1.
        if let Some(saved) = session::load() {
            if saved.book != self.session.book || saved.chapter != self.session.chapter {
                self.session.part = 1;
                self.session.timeline_pos = 0.0;
            }
        } else {
            self.session.part = 1;
            self.session.timeline_pos = 0.0;
        }

        let _ = session::save(&self.session);
        self.setup_error = None;
        self.mode = AppMode::Ready;
    }

    fn handle_key_ready(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char(' ') | KeyCode::Enter => self.begin_precheck(),
            KeyCode::Char('e') | KeyCode::Char('E') => {
                self.setup_book = self.session.book.clone();
                self.setup_chapter = format!("{:02}", self.session.chapter);
                self.setup_field = SetupField::Book;
                self.setup_error = None;
                self.mode = AppMode::Setup;
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                self.detect_mic();
            }
            KeyCode::Char('q') | KeyCode::Esc => {
                self.should_quit = true;
            }
            _ => {}
        }
    }

    fn handle_key_precheck(&mut self, key: KeyEvent) {
        // Allow aborting the pre-check.
        if matches!(key.code, KeyCode::Esc | KeyCode::Char('q')) {
            self.abort_precheck();
        }
    }

    fn handle_key_recording(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char(' ') | KeyCode::Enter => self.stop_recording(),
            KeyCode::Char('p') | KeyCode::Char('P') => self.begin_punch(),
            _ => {}
        }
    }

    fn handle_key_post(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                // Continue same chapter — next part.
                self.session.advance_part();
                let _ = session::save(&self.session);
                self.status_msg = Some(format!(
                    "Saved. Ready for part {}.",
                    self.session.part
                ));
                self.mode = AppMode::Ready;
            }
            KeyCode::Char('n') | KeyCode::Char('N') => {
                // Chapter complete.
                self.session.advance_chapter(); // also resets timeline_pos
                let _ = session::save(&self.session);
                self.status_msg = Some(format!(
                    "Chapter complete. Ready for Chapter {:02}.",
                    self.session.chapter
                ));
                self.mode = AppMode::Ready;
            }
            _ => {}
        }
    }

    fn handle_key_punch_rollback(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char(' ') | KeyCode::Enter => self.punch_in(),
            KeyCode::Esc | KeyCode::Char('q') => self.abort_punch_rollback(),
            _ => {}
        }
    }

    fn handle_key_mic_error(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('r') | KeyCode::Char('R') | KeyCode::Enter => {
                self.detect_mic();
            }
            KeyCode::Char('q') | KeyCode::Esc => {
                self.should_quit = true;
            }
            _ => {}
        }
    }

    // ── Recording lifecycle ───────────────────────────────────────────────────

    fn begin_precheck(&mut self) {
        // Re-verify the mic is still present before opening it.
        if let Some(ref mic) = self.mic {
            if !device::is_mic_present(mic) {
                self.error_msg = "Microphone disconnected.".into();
                self.mic = None;
                self.mode = AppMode::MicError;
                return;
            }
        } else {
            self.detect_mic();
            if self.mode == AppMode::MicError {
                return;
            }
        }

        // Skip the signal check after the first successful pre-check.
        if self.precheck_done {
            self.start_recording();
            return;
        }

        let alsa_name = self.mic.as_ref().unwrap().alsa_name.clone();

        match audio::start_capture(&alsa_name) {
            Ok(handle) => {
                self.capture = Some(handle);
                self.precheck_frames_seen = 0;
                self.precheck_signal_found = false;
                self.mode = AppMode::PreCheck;
            }
            Err(e) => {
                self.error_msg = format!("Cannot open ALSA device: {}", e);
                self.mode = AppMode::Fatal;
            }
        }
    }

    fn abort_precheck(&mut self) {
        if let Some(handle) = self.capture.take() {
            handle.stop();
        }
        self.mode = AppMode::Ready;
        self.status_msg = Some("Signal check aborted.".into());
    }

    fn start_recording(&mut self) {
        // Re-open ALSA for recording (a fresh capture session).
        let alsa_name = self.mic.as_ref().unwrap().alsa_name.clone();

        match audio::start_capture(&alsa_name) {
            Ok(handle) => {
                // Prepare the output file.
                if let Err(e) = self.session.ensure_output_dir() {
                    self.error_msg = format!("Cannot create output directory: {}", e);
                    handle.stop();
                    self.mode = AppMode::Fatal;
                    return;
                }

                let path = self.session.next_free_path();

                match WavWriter::new(&path) {
                    Ok(wav) => {
                        // Snapshot the clip's timeline start position.
                        self.clip_start_timeline = self.session.timeline_pos;

                        // Append timeline entry now (before audio is written)
                        // so a crash still leaves a valid record.
                        let filename = path
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        if let Err(e) = self.session.append_timeline_entry(
                            &filename,
                            self.clip_start_timeline,
                        ) {
                            // Non-fatal: warn but do not abort recording.
                            self.status_msg =
                                Some(format!("Warning: could not write timeline: {}", e));
                        }

                        self.active_file = Some(path);
                        self.wav = Some(wav);
                        self.capture = Some(handle);
                        self.last_rms_db = -100.0;
                        self.silence_warning = false;
                        self.last_non_silence = Instant::now();
                        self.record_start = Some(Instant::now());
                        self.mode = AppMode::Recording;
                    }
                    Err(e) => {
                        self.error_msg = format!("Cannot create WAV file: {}", e);
                        handle.stop();
                        self.mode = AppMode::Fatal;
                    }
                }
            }
            Err(e) => {
                self.error_msg = format!("Cannot open ALSA device: {}", e);
                self.mode = AppMode::Fatal;
            }
        }
    }

    fn stop_recording(&mut self) {
        // Capture clip duration before dropping the WavWriter.
        let duration = self
            .wav
            .as_ref()
            .map(|w| Session::duration_from_bytes(w.data_bytes()))
            .unwrap_or(0.0);

        if let Some(handle) = self.capture.take() {
            handle.stop();
        }
        if let Some(mut wav) = self.wav.take() {
            if let Err(e) = wav.finalize() {
                self.error_msg = format!("WAV finalise error: {}", e);
                self.mode = AppMode::Fatal;
                return;
            }
        }

        // Discard clips that are too short — delete the file and retract the
        // timeline entry that was written at recording start.
        if self.discard_short_clips && duration < self.discard_duration_secs {
            if let Some(ref path) = self.active_file {
                let _ = fs::remove_file(path);
            }
            let _ = self.session.remove_last_timeline_entry();
            // Timeline does not advance — the clip never happened.
            let _ = session::save(&self.session);
            self.active_file = None;
            self.record_start = None;
            self.status_msg = Some(format!(
                "Clip discarded (shorter than {:.1}s threshold).",
                self.discard_duration_secs
            ));
            self.mode = AppMode::PostRecording;
            return;
        }

        // Advance timeline past the end of this clip.
        self.session.timeline_pos = self.clip_start_timeline + duration;
        let _ = session::save(&self.session);

        self.record_start = None;
        self.mode = AppMode::PostRecording;
    }

    // ── Punch-and-roll lifecycle ──────────────────────────────────────────────

    /// Called when P is pressed during Recording.  Stops the current recording,
    /// computes the rollback point, and starts playing back the last N seconds.
    fn begin_punch(&mut self) {
        // Capture clip duration before dropping the WavWriter.
        let duration = self
            .wav
            .as_ref()
            .map(|w| Session::duration_from_bytes(w.data_bytes()))
            .unwrap_or(0.0);
        self.last_clip_duration = duration;

        if let Some(handle) = self.capture.take() {
            handle.stop();
        }
        if let Some(mut wav) = self.wav.take() {
            if let Err(e) = wav.finalize() {
                self.error_msg = format!("WAV finalise error during punch: {}", e);
                self.mode = AppMode::Fatal;
                return;
            }
        }
        self.record_start = None;

        // Discard clips that are too short — can't punch-roll on them meaningfully.
        // Retract the clip, restore timeline, and return to PostRecording.
        if self.discard_short_clips && self.last_clip_duration < self.discard_duration_secs {
            if let Some(ref path) = self.active_file {
                let _ = fs::remove_file(path);
            }
            let _ = self.session.remove_last_timeline_entry();
            self.session.timeline_pos = self.clip_start_timeline;
            let _ = session::save(&self.session);
            self.active_file = None;
            self.status_msg = Some(format!(
                "Clip discarded (shorter than {:.1}s threshold).",
                self.discard_duration_secs
            ));
            self.mode = AppMode::PostRecording;
            return;
        }

        // Compute rollback: rewind punch_back_time seconds from the end of the
        // clip (clamped so we never seek before the clip's own start).
        let rollback_offset = (duration - self.punch_back_time).max(0.0);
        self.punch_rollback_abs = self.clip_start_timeline + rollback_offset;

        let path = match &self.active_file {
            Some(p) => p.clone(),
            None => {
                self.error_msg = "No active file for punch-and-roll.".into();
                self.mode = AppMode::Fatal;
                return;
            }
        };

        match pb::start_playback(path, rollback_offset) {
            Ok(handle) => {
                self.playback = Some(handle);
                self.playback_start = Some(Instant::now());
                self.playback_done = false;
                self.playback_elapsed = 0.0;
                self.mode = AppMode::PunchRollback;
            }
            Err(e) => {
                self.error_msg = format!("Cannot start playback: {}", e);
                self.mode = AppMode::Fatal;
            }
        }
    }

    /// Called when Space is pressed during PunchRollback.  Stops playback,
    /// sets the timeline to the current playback position (overlapping the
    /// previous clip), and starts recording immediately.
    fn punch_in(&mut self) {
        if let Some(h) = self.playback.take() {
            h.signal_stop(); // Non-blocking: don't freeze the main thread on writei
        }

        let elapsed = if self.playback_done {
            self.playback_elapsed
        } else {
            self.playback_start
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0)
        };

        // Set timeline to exactly where playback was when Space was pressed.
        // Capped at the old clip's end so we never create a gap.
        let clip_end = self.clip_start_timeline + self.last_clip_duration;
        let punch_in_time = (self.punch_rollback_abs + elapsed).min(clip_end);

        self.session.timeline_pos = punch_in_time;
        self.session.advance_part();
        let _ = session::save(&self.session);

        self.start_recording();
    }

    /// Called when Esc is pressed during PunchRollback.
    fn abort_punch_rollback(&mut self) {
        if let Some(h) = self.playback.take() {
            h.stop();
        }
        // The clip we just recorded is complete; advance timeline past it.
        self.session.timeline_pos = self.clip_start_timeline + self.last_clip_duration;
        let _ = session::save(&self.session);

        self.mode = AppMode::PostRecording;
        self.status_msg = Some("Punch-and-roll cancelled.".into());
    }

    // ── Periodic tick ─────────────────────────────────────────────────────────

    /// Called by the main loop on every iteration to drain the audio channel
    /// and advance time-based state.
    pub fn tick(&mut self) {
        // Periodically re-check mic presence even in Ready mode.
        if self.mode == AppMode::Ready {
            if let Some(ref mic) = self.mic.clone() {
                if !device::is_mic_present(mic) {
                    self.error_msg = "Microphone disconnected.".into();
                    self.mic = None;
                    self.mode = AppMode::MicError;
                    return;
                }
            }
        }

        match self.mode {
            AppMode::PreCheck => self.tick_precheck(),
            AppMode::Recording => self.tick_recording(),
            AppMode::PunchRollback => self.tick_punch_rollback(),
            _ => {}
        }
    }

    fn tick_precheck(&mut self) {
        // Phase 1 — collect events without holding a mutable borrow on self.
        let events = self.collect_audio_events();
        if events.is_empty() {
            return;
        }

        // Phase 2 — process events; self is fully mutable here.
        for event in events {
            if !matches!(self.mode, AppMode::PreCheck) {
                break;
            }
            match event {
                AudioEvent::Samples { data, rms_db } => {
                    self.precheck_frames_seen += data.len();
                    self.last_rms_db = rms_db;
                    if rms_db > SILENCE_THRESHOLD_DB {
                        self.precheck_signal_found = true;
                        self.activity_tick = self.activity_tick.wrapping_add(1);
                    }
                    if self.precheck_frames_seen >= self.precheck_target_frames {
                        self.finish_precheck();
                        return;
                    }
                }
                AudioEvent::DeviceGone => {
                    self.abort_precheck_device_gone();
                    return;
                }
                AudioEvent::Xrun => {
                    // xrun during pre-check: reset window and continue.
                    self.precheck_frames_seen = 0;
                    self.precheck_signal_found = false;
                }
                AudioEvent::Error(msg) => {
                    if let Some(h) = self.capture.take() {
                        h.stop();
                    }
                    self.error_msg = format!("ALSA error: {}", msg);
                    self.mode = AppMode::Fatal;
                    return;
                }
            }
        }
    }

    fn finish_precheck(&mut self) {
        // Stop the pre-check capture.
        if let Some(h) = self.capture.take() {
            h.stop();
        }

        if self.precheck_signal_found {
            self.precheck_done = true;
            self.start_recording();
        } else {
            self.status_msg = Some(
                "No audio detected — check the microphone mute button.".into(),
            );
            self.mode = AppMode::Ready;
        }
    }

    fn abort_precheck_device_gone(&mut self) {
        if let Some(h) = self.capture.take() {
            h.stop();
        }
        self.error_msg = "Microphone disconnected during signal check.".into();
        self.mic = None;
        self.mode = AppMode::MicError;
    }

    fn tick_recording(&mut self) {
        // Phase 1 — collect events (immutable borrow ends before processing).
        let (events, disconnected) = self.collect_audio_events_with_disconnect();

        // Phase 2 — process events.
        for event in events {
            if !matches!(self.mode, AppMode::Recording) {
                return;
            }
            match event {
                AudioEvent::Samples { data, rms_db } => {
                    if let Some(ref mut wav) = self.wav {
                        if let Err(e) = wav.write_s24le(&data) {
                            let _ = wav.finalize();
                            self.wav = None;
                            if let Some(h) = self.capture.take() {
                                h.stop();
                            }
                            self.error_msg = format!("Disk write error: {}", e);
                            self.mode = AppMode::Fatal;
                            return;
                        }
                    }

                    self.last_rms_db = rms_db;

                    if rms_db > SILENCE_THRESHOLD_DB {
                        self.last_non_silence = Instant::now();
                        self.activity_tick = self.activity_tick.wrapping_add(1);
                        self.silence_warning = false;
                    } else {
                        let silent_for = self.last_non_silence.elapsed().as_secs_f64();
                        if silent_for >= SILENCE_WARNING_SECONDS {
                            self.silence_warning = true;
                        }
                    }
                }
                AudioEvent::Xrun => {
                    self.status_msg = Some("WARNING: audio buffer overrun (xrun).".into());
                }
                AudioEvent::DeviceGone => {
                    if let Some(mut wav) = self.wav.take() {
                        let _ = wav.finalize();
                    }
                    if let Some(h) = self.capture.take() {
                        h.stop();
                    }
                    self.error_msg =
                        "Microphone unplugged — recording stopped. File saved.".into();
                    self.mic = None;
                    self.record_start = None;
                    self.mode = AppMode::MicError;
                    return;
                }
                AudioEvent::Error(msg) => {
                    if let Some(mut wav) = self.wav.take() {
                        let _ = wav.finalize();
                    }
                    if let Some(h) = self.capture.take() {
                        h.stop();
                    }
                    self.error_msg = format!("ALSA error during recording: {}", msg);
                    self.record_start = None;
                    self.mode = AppMode::Fatal;
                    return;
                }
            }
        }

        if disconnected && matches!(self.mode, AppMode::Recording) {
            if let Some(mut wav) = self.wav.take() {
                let _ = wav.finalize();
            }
            self.capture = None;
            self.error_msg = "Audio thread exited unexpectedly.".into();
            self.record_start = None;
            self.mode = AppMode::Fatal;
        }
    }

    fn tick_punch_rollback(&mut self) {
        // Drain playback events.
        let mut events = Vec::new();
        if let Some(ref h) = self.playback {
            loop {
                match h.rx.try_recv() {
                    Ok(e) => events.push(e),
                    Err(_) => break,
                }
            }
        }

        for event in events {
            match event {
                PlaybackEvent::Done => {
                    if !self.playback_done {
                        self.playback_done = true;
                        // Freeze elapsed at the natural end of the clip segment.
                        self.playback_elapsed = self
                            .playback_start
                            .map(|t| t.elapsed().as_secs_f64())
                            .unwrap_or(0.0);
                    }
                }
                PlaybackEvent::Error(msg) => {
                    self.playback = None;
                    self.error_msg = format!("Playback error: {}", msg);
                    self.mode = AppMode::Fatal;
                    return;
                }
            }
        }
    }

    // ── Clean shutdown ────────────────────────────────────────────────────────

    /// Best-effort clean shutdown for SIGTERM / Ctrl-C.
    pub fn emergency_stop(&mut self) {
        if let Some(mut wav) = self.wav.take() {
            let _ = wav.finalize();
        }
        if let Some(h) = self.capture.take() {
            h.stop();
        }
        if let Some(h) = self.playback.take() {
            h.stop();
        }
        let _ = session::save(&self.session);
        self.should_quit = true;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    pub fn elapsed_recording(&self) -> Option<Duration> {
        self.record_start.map(|t| t.elapsed())
    }

    /// Current playback elapsed seconds (live or frozen at Done).
    pub fn playback_elapsed_secs(&self) -> f64 {
        if self.playback_done {
            self.playback_elapsed
        } else {
            self.playback_start
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0)
        }
    }

    /// Drain all pending audio events from the channel into a Vec.
    fn collect_audio_events(&self) -> Vec<AudioEvent> {
        let mut events = Vec::new();
        if let Some(ref h) = self.capture {
            loop {
                match h.rx.try_recv() {
                    Ok(e) => events.push(e),
                    Err(_) => break,
                }
            }
        }
        events
    }

    /// Like `collect_audio_events` but also returns a bool indicating whether
    /// the sender side has disconnected (channel closed).
    fn collect_audio_events_with_disconnect(&self) -> (Vec<AudioEvent>, bool) {
        let mut events = Vec::new();
        let mut disconnected = false;
        if let Some(ref h) = self.capture {
            loop {
                match h.rx.try_recv() {
                    Ok(e) => events.push(e),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        (events, disconnected)
    }
}

