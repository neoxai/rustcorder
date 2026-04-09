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
    /// Idle — cursor parked at a timeline position.
    Standby,
    /// Streaming audio from the timeline.
    Playing,
    /// Actively capturing audio to disk.
    Recording,
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

    // ── Live capture resources (Some when recording or during startup check)
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

    // ── Startup mic check state ───────────────────────────────────────────
    pub precheck_frames_seen: usize,
    pub precheck_target_frames: usize,
    /// True once the one-time startup mic check has completed successfully.
    pub precheck_done: bool,

    // ── Recording timing ──────────────────────────────────────────────────
    pub record_start: Option<Instant>,

    // ── Playback ──────────────────────────────────────────────────────────
    /// How far back (seconds) to rewind on P. Loaded from PUNCH_BACK_TIME.
    pub punch_back_time: f64,
    /// Absolute timeline position (seconds) when the current/last clip started.
    pub clip_start_timeline: f64,
    /// Duration (seconds) of the clip that triggered a punch operation.
    pub last_clip_duration: f64,
    /// Absolute timeline position where playback began (the "playhead origin").
    pub playhead_origin: f64,
    /// True when we entered Playing mode by pressing P during Recording
    /// (as opposed to L from Standby).  Used to decide whether to advance
    /// the part counter when playback ends or is interrupted.
    pub punch_play: bool,
    /// Active playback thread handle (Some during Playing).
    pub playback: Option<PlaybackHandle>,
    /// When playback started (used to compute live playhead position).
    pub playback_start: Option<Instant>,
    /// True once the playback thread has signalled PlaybackEvent::Done.
    pub playback_done: bool,
    /// Elapsed seconds frozen at the moment PlaybackEvent::Done arrived.
    pub playback_elapsed: f64,

    // ── Status / error messages ───────────────────────────────────────────
    pub status_msg: Option<String>,
    pub error_msg: String,

    // ── Short-clip discard ────────────────────────────────────────────────
    pub discard_short_clips: bool,
    pub discard_duration_secs: f64,

    // ── EPUB path (for browser viewer) ───────────────────────────────────
    pub epub_path: Option<std::path::PathBuf>,

    // ── Quit flag ─────────────────────────────────────────────────────────
    pub should_quit: bool,
}

impl App {
    pub fn new() -> Self {
        let session = session::load().unwrap_or_else(|| Session::new(String::new(), 1));
        let mode = if session.book.is_empty() {
            AppMode::Setup
        } else {
            AppMode::Standby
        };

        let setup_book = session.book.clone();
        let setup_chapter = format!("{:02}", session.chapter);

        let punch_back_time = std::env::var("PUNCH_BACK_TIME")
            .ok()
            .and_then(|v| v.trim().trim_end_matches('s').parse::<f64>().ok())
            .unwrap_or(15.0)
            .max(1.0);

        let discard_short_clips = std::env::var("DISCARD_SHORT_CLIPS")
            .map(|v| v.trim().eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let discard_duration_secs = std::env::var("DISCARD_DURATION")
            .ok()
            .and_then(|v| v.trim().trim_end_matches('s').parse::<f64>().ok())
            .unwrap_or(1.0)
            .max(0.0);

        let epub_path = if !session.book.is_empty() {
            find_epub_in_dir(&session.output_dir())
        } else {
            None
        };

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
            precheck_target_frames: (PRECHECK_SECONDS * SAMPLE_RATE as f64) as usize,
            precheck_done: false,
            record_start: None,
            punch_back_time,
            clip_start_timeline: 0.0,
            last_clip_duration: 0.0,
            playhead_origin: 0.0,
            punch_play: false,
            playback: None,
            playback_start: None,
            playback_done: false,
            playback_elapsed: 0.0,
            status_msg: None,
            error_msg: String::new(),
            discard_short_clips,
            discard_duration_secs,
            epub_path,
            should_quit: false,
        }
    }

    // ── Device detection ──────────────────────────────────────────────────────

    pub fn detect_mic(&mut self) {
        match device::find_approved_mic() {
            Ok(mic) => {
                self.mic = Some(mic);
                // Only recover from MicError to Standby; don't override Setup.
                if self.mode == AppMode::MicError {
                    self.mode = AppMode::Standby;
                }
            }
            Err(e) => {
                self.mic = None;
                self.error_msg = e.to_string();
                self.mode = AppMode::MicError;
            }
        }
    }

    // ── EPUB path refresh ─────────────────────────────────────────────────────

    fn refresh_epub_path(&mut self) {
        self.epub_path = find_epub_in_dir(&self.session.output_dir());
    }

    // ── Keyboard handling ─────────────────────────────────────────────────────

    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && key.code == KeyCode::Char('c')
        {
            self.emergency_stop();
            return true;
        }

        match self.mode {
            AppMode::Setup     => self.handle_key_setup(key),
            AppMode::Standby   => self.handle_key_standby(key),
            AppMode::Playing   => self.handle_key_playing(key),
            AppMode::Recording => self.handle_key_recording(key),
            AppMode::MicError  => self.handle_key_mic_error(key),
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
                    SetupField::Book    => SetupField::Chapter,
                    SetupField::Chapter => SetupField::Book,
                };
            }
            KeyCode::Enter => self.confirm_setup(),
            KeyCode::Esc => {
                if !self.session.book.is_empty() {
                    self.setup_error = None;
                    self.mode = AppMode::Standby;
                }
            }
            KeyCode::Backspace => match self.setup_field {
                SetupField::Book    => { self.setup_book.pop(); }
                SetupField::Chapter => { self.setup_chapter.pop(); }
            },
            KeyCode::Char(c) => match self.setup_field {
                SetupField::Book => {
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
        self.refresh_epub_path();
        self.mode = AppMode::Standby;
    }

    fn handle_key_standby(&mut self, key: KeyEvent) {
        // Abort background mic check on any action key so it doesn't interfere.
        match key.code {
            KeyCode::Char(' ') | KeyCode::Enter => {
                self.abort_background_check();
                self.start_recording();
            }
            KeyCode::Char('l') | KeyCode::Char('L') => {
                self.abort_background_check();
                let pos = self.session.timeline_pos;
                self.begin_playing(pos);
            }
            KeyCode::Char('j') | KeyCode::Char('J') => self.standby_jump_back(),
            KeyCode::Char('k') | KeyCode::Char('K') => self.standby_jump_forward(),
            KeyCode::Char('n') | KeyCode::Char('N') => self.advance_chapter_cmd(),
            KeyCode::Char('e') | KeyCode::Char('E') => {
                self.setup_book = self.session.book.clone();
                self.setup_chapter = format!("{:02}", self.session.chapter);
                self.setup_field = SetupField::Book;
                self.setup_error = None;
                self.mode = AppMode::Setup;
            }
            KeyCode::Char('r') | KeyCode::Char('R') => self.detect_mic(),
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            _ => {}
        }
    }

    fn handle_key_playing(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char(' ') | KeyCode::Enter  => self.playing_punch_in(),
            KeyCode::Char('l') | KeyCode::Char('L') => self.playing_to_standby(),
            KeyCode::Char('p') | KeyCode::Char('P') => self.rewind_playing(),
            KeyCode::Char('j') | KeyCode::Char('J') => self.playing_jump_back(),
            KeyCode::Char('k') | KeyCode::Char('K') => self.playing_jump_forward(),
            _ => {}
        }
    }

    fn handle_key_recording(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char(' ') | KeyCode::Enter => self.stop_recording_to_standby(),
            KeyCode::Char('p') | KeyCode::Char('P') => self.punch_to_playing(),
            _ => {}
        }
    }

    fn handle_key_mic_error(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('r') | KeyCode::Char('R') | KeyCode::Enter => self.detect_mic(),
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            _ => {}
        }
    }

    // ── Startup background mic check ──────────────────────────────────────────

    /// Start the one-time 2-second background mic check.
    /// Called on the first Standby tick after mic detection.
    fn start_background_mic_check(&mut self) {
        let Some(ref mic) = self.mic else { return; };
        let alsa_name = mic.alsa_name.clone();
        match audio::start_capture(&alsa_name) {
            Ok(handle) => {
                self.capture = Some(handle);
                self.precheck_frames_seen = 0;
            }
            Err(_) => {
                // Silently skip — mic presence was already confirmed by detect_mic.
            }
        }
    }

    /// Stop and discard the background check if still running.
    fn abort_background_check(&mut self) {
        if !self.precheck_done {
            if let Some(h) = self.capture.take() {
                h.stop();
            }
        }
    }

    // ── Recording lifecycle ───────────────────────────────────────────────────

    fn start_recording(&mut self) {
        // Verify mic is still present.
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

        let alsa_name = self.mic.as_ref().unwrap().alsa_name.clone();

        match audio::start_capture(&alsa_name) {
            Ok(handle) => {
                if let Err(e) = self.session.ensure_output_dir() {
                    self.error_msg = format!("Cannot create output directory: {}", e);
                    handle.stop();
                    self.mode = AppMode::Fatal;
                    return;
                }

                let path = self.session.next_free_path();

                match WavWriter::new(&path) {
                    Ok(wav) => {
                        self.clip_start_timeline = self.session.timeline_pos;

                        let filename = path
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        if let Err(e) = self.session.append_timeline_entry(
                            &filename,
                            self.clip_start_timeline,
                        ) {
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

    /// Stop recording and return to Standby.
    fn stop_recording_to_standby(&mut self) {
        let duration = self.wav.as_ref()
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

        if self.discard_short_clips && duration < self.discard_duration_secs {
            if let Some(ref path) = self.active_file {
                let _ = fs::remove_file(path);
            }
            let _ = self.session.remove_last_timeline_entry();
            let _ = session::save(&self.session);
            self.active_file = None;
            self.record_start = None;
            self.status_msg = Some(format!(
                "Clip discarded (shorter than {:.1}s threshold).",
                self.discard_duration_secs
            ));
            self.mode = AppMode::Standby;
            return;
        }

        self.session.timeline_pos = self.clip_start_timeline + duration;
        self.session.advance_part();
        let _ = session::save(&self.session);
        self.record_start = None;
        self.active_file = None;
        self.mode = AppMode::Standby;
    }

    // ── Playback lifecycle ────────────────────────────────────────────────────

    /// Start playing from `from_pos` using the chapter timeline.
    /// Used for Standby → Playing (L) and all J/K navigation.
    fn begin_playing(&mut self, from_pos: f64) {
        self.playhead_origin = from_pos;

        let timeline_path = self.session.output_dir()
            .join(format!("Chapter_{:02}_timeline.txt", self.session.chapter));

        if !timeline_path.exists() {
            self.status_msg = Some("No audio recorded yet for this chapter.".into());
            return;
        }

        let segs = match crate::export::timeline::timeline_segments_from(
            &timeline_path,
            &self.session.output_dir(),
            from_pos,
        ) {
            Ok(s) => s,
            Err(e) => {
                self.status_msg = Some(format!("Cannot read timeline: {}", e));
                return;
            }
        };

        if segs.is_empty() {
            self.status_msg = Some("No audio to play from this position.".into());
            return;
        }

        let segments: Vec<pb::PlaybackSegment> = segs.into_iter().map(|s| pb::PlaybackSegment {
            path: s.path,
            offset_secs: s.file_offset_secs,
            limit_secs: s.play_secs,
        }).collect();

        match pb::start_playback_segments(segments) {
            Ok(handle) => {
                self.playback = Some(handle);
                self.playback_start = Some(Instant::now());
                self.playback_done = false;
                self.playback_elapsed = 0.0;
                self.mode = AppMode::Playing;
            }
            Err(e) => {
                self.error_msg = format!("Cannot start playback: {}", e);
                self.mode = AppMode::Fatal;
            }
        }
    }

    /// L pressed during Playing: stop and park at current playhead.
    fn playing_to_standby(&mut self) {
        if let Some(h) = self.playback.take() {
            h.stop();
        }
        if self.punch_play {
            // Came from a punch — advance timeline past the recorded clip
            // and increment the part counter (clip was saved but no punch-in).
            self.session.timeline_pos = self.clip_start_timeline + self.last_clip_duration;
            self.session.advance_part();
            self.active_file = None;
            self.punch_play = false;
        } else {
            let elapsed = self.playback_elapsed_secs();
            self.session.timeline_pos = self.playhead_origin + elapsed;
        }
        let _ = session::save(&self.session);
        self.mode = AppMode::Standby;
    }

    /// Space pressed during Playing: stop playback and start recording at
    /// current playhead position (punch-in).
    fn playing_punch_in(&mut self) {
        if let Some(h) = self.playback.take() {
            h.signal_stop();
        }

        let elapsed = self.playback_elapsed_secs();
        let punch_in_time = self.playhead_origin + elapsed;

        self.session.timeline_pos = punch_in_time;

        // If we came from a punch (Recording → P), the just-recorded clip
        // is already at session.part-1; advance now so the new take gets a
        // fresh part number.
        if self.punch_play {
            self.session.advance_part();
            self.active_file = None;
            self.punch_play = false;
        }

        let _ = session::save(&self.session);
        self.start_recording();
    }

    /// P pressed during Playing: rewind punch_back_time and restart.
    fn rewind_playing(&mut self) {
        if let Some(h) = self.playback.take() {
            h.stop();
        }
        let new_origin = (self.playhead_origin - self.punch_back_time).max(0.0);
        self.begin_playing(new_origin);
    }

    /// J pressed during Playing: jump to start of current part and restart.
    fn playing_jump_back(&mut self) {
        let current_pos = self.playhead_origin + self.playback_elapsed_secs();
        let timeline_path = self.session.output_dir()
            .join(format!("Chapter_{:02}_timeline.txt", self.session.chapter));
        let new_pos = match crate::export::timeline::current_part_start(
            &timeline_path, &self.session.output_dir(), current_pos) {
            Ok(s) => s,
            Err(_) => return,
        };
        if let Some(h) = self.playback.take() { h.stop(); }
        self.begin_playing(new_pos);
    }

    /// K pressed during Playing: jump to start of next part and restart.
    fn playing_jump_forward(&mut self) {
        let current_pos = self.playhead_origin + self.playback_elapsed_secs();
        let timeline_path = self.session.output_dir()
            .join(format!("Chapter_{:02}_timeline.txt", self.session.chapter));
        let new_pos = match crate::export::timeline::next_part_start(
            &timeline_path, &self.session.output_dir(), current_pos) {
            Ok(Some(n)) => n,
            Ok(None) => {
                // Already at last part — jump to chapter end.
                match crate::export::timeline::chapter_end(
                    &timeline_path, &self.session.output_dir()) {
                    Ok(end) => end,
                    Err(_) => return,
                }
            }
            Err(_) => return,
        };
        if let Some(h) = self.playback.take() { h.stop(); }
        self.begin_playing(new_pos);
    }

    // ── Punch-and-roll: Recording → Playing ───────────────────────────────────

    /// P pressed during Recording: finalize the clip, rewind, enter Playing.
    fn punch_to_playing(&mut self) {
        let duration = self.wav.as_ref()
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
            self.mode = AppMode::Standby;
            return;
        }

        let clip_end = self.clip_start_timeline + duration;
        let origin = (clip_end - self.punch_back_time).max(0.0);

        self.punch_play = true;
        // active_file stays set so playing_to_standby / playing_punch_in know
        // a clip was just recorded.
        self.begin_playing(origin);
    }

    // ── Standby navigation (J / K / N) ───────────────────────────────────────

    fn standby_jump_back(&mut self) {
        let timeline_path = self.session.output_dir()
            .join(format!("Chapter_{:02}_timeline.txt", self.session.chapter));
        match crate::export::timeline::current_part_start(
            &timeline_path, &self.session.output_dir(), self.session.timeline_pos) {
            Ok(start) => {
                self.session.timeline_pos = start;
                let _ = session::save(&self.session);
            }
            Err(_) => {} // no timeline yet
        }
    }

    fn standby_jump_forward(&mut self) {
        let timeline_path = self.session.output_dir()
            .join(format!("Chapter_{:02}_timeline.txt", self.session.chapter));
        match crate::export::timeline::next_part_start(
            &timeline_path, &self.session.output_dir(), self.session.timeline_pos) {
            Ok(Some(next)) => {
                self.session.timeline_pos = next;
                let _ = session::save(&self.session);
            }
            Ok(None) => {
                if let Ok(end) = crate::export::timeline::chapter_end(
                    &timeline_path, &self.session.output_dir()) {
                    self.session.timeline_pos = end;
                    let _ = session::save(&self.session);
                }
            }
            Err(_) => {}
        }
    }

    fn advance_chapter_cmd(&mut self) {
        self.session.advance_chapter();
        let _ = session::save(&self.session);
        self.refresh_epub_path();
        self.status_msg = Some(format!(
            "Advanced to Chapter {:02}.",
            self.session.chapter
        ));
    }

    // ── Periodic tick ─────────────────────────────────────────────────────────

    pub fn tick(&mut self) {
        // Periodically re-check mic presence in Standby.
        if self.mode == AppMode::Standby {
            if let Some(ref mic) = self.mic.clone() {
                if !device::is_mic_present(mic) {
                    self.error_msg = "Microphone disconnected.".into();
                    self.mic = None;
                    if let Some(h) = self.capture.take() { h.stop(); } // abort bg check
                    self.mode = AppMode::MicError;
                    return;
                }
            }
        }

        match self.mode {
            AppMode::Standby   => self.tick_standby(),
            AppMode::Recording => self.tick_recording(),
            AppMode::Playing   => self.tick_playing(),
            _ => {}
        }
    }

    fn tick_standby(&mut self) {
        // Start one-time background mic check on first Standby tick.
        if !self.precheck_done && self.capture.is_none() && self.mic.is_some() {
            self.start_background_mic_check();
        }

        // Drain background check events (non-blocking).
        if self.capture.is_some() && !self.precheck_done {
            let events = self.collect_audio_events();
            for event in events {
                match event {
                    AudioEvent::Samples { data, rms_db } => {
                        self.precheck_frames_seen += data.len();
                        self.last_rms_db = rms_db;
                        if self.precheck_frames_seen >= self.precheck_target_frames {
                            if let Some(h) = self.capture.take() { h.stop(); }
                            self.precheck_done = true;
                            return;
                        }
                    }
                    AudioEvent::DeviceGone => {
                        if let Some(h) = self.capture.take() { h.stop(); }
                        self.error_msg = "Microphone disconnected during startup check.".into();
                        self.mic = None;
                        self.mode = AppMode::MicError;
                        return;
                    }
                    AudioEvent::Error(msg) => {
                        if let Some(h) = self.capture.take() { h.stop(); }
                        self.error_msg = format!("ALSA error: {}", msg);
                        self.mode = AppMode::Fatal;
                        return;
                    }
                    AudioEvent::Xrun => {
                        self.precheck_frames_seen = 0;
                    }
                }
            }
        }
    }

    fn tick_recording(&mut self) {
        let (events, disconnected) = self.collect_audio_events_with_disconnect();

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
                            if let Some(h) = self.capture.take() { h.stop(); }
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
                    if let Some(mut wav) = self.wav.take() { let _ = wav.finalize(); }
                    if let Some(h) = self.capture.take() { h.stop(); }
                    self.error_msg =
                        "Microphone unplugged — recording stopped. File saved.".into();
                    self.mic = None;
                    self.record_start = None;
                    self.mode = AppMode::MicError;
                    return;
                }
                AudioEvent::Error(msg) => {
                    if let Some(mut wav) = self.wav.take() { let _ = wav.finalize(); }
                    if let Some(h) = self.capture.take() { h.stop(); }
                    self.error_msg = format!("ALSA error during recording: {}", msg);
                    self.record_start = None;
                    self.mode = AppMode::Fatal;
                    return;
                }
            }
        }

        if disconnected && matches!(self.mode, AppMode::Recording) {
            if let Some(mut wav) = self.wav.take() { let _ = wav.finalize(); }
            self.capture = None;
            self.error_msg = "Audio thread exited unexpectedly.".into();
            self.record_start = None;
            self.mode = AppMode::Fatal;
        }
    }

    fn tick_playing(&mut self) {
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
                        self.playback_elapsed = self.playback_start
                            .map(|t| t.elapsed().as_secs_f64())
                            .unwrap_or(0.0);

                        // Auto-transition to Standby at end of audio.
                        let end_pos = self.playhead_origin + self.playback_elapsed;
                        self.session.timeline_pos = end_pos;
                        if self.punch_play {
                            self.session.advance_part();
                            self.active_file = None;
                            self.punch_play = false;
                        }
                        let _ = session::save(&self.session);
                        self.mode = AppMode::Standby;
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

    pub fn emergency_stop(&mut self) {
        if let Some(mut wav) = self.wav.take() { let _ = wav.finalize(); }
        if let Some(h) = self.capture.take() { h.stop(); }
        if let Some(h) = self.playback.take() { h.stop(); }
        let _ = session::save(&self.session);
        self.should_quit = true;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    pub fn elapsed_recording(&self) -> Option<Duration> {
        self.record_start.map(|t| t.elapsed())
    }

    /// Current playback elapsed seconds (live, or frozen at Done).
    pub fn playback_elapsed_secs(&self) -> f64 {
        if self.playback_done {
            self.playback_elapsed
        } else {
            self.playback_start
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0)
        }
    }

    /// Current playhead position in absolute timeline seconds.
    /// During Playing this is live; otherwise it equals timeline_pos.
    pub fn playhead_pos(&self) -> f64 {
        if self.mode == AppMode::Playing {
            self.playhead_origin + self.playback_elapsed_secs()
        } else {
            self.session.timeline_pos
        }
    }

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

// ── EPUB discovery ────────────────────────────────────────────────────────────

fn find_epub_in_dir(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().map(|e| e == "epub").unwrap_or(false) {
            return Some(path);
        }
    }
    None
}
