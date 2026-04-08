pub mod cpal_player;
pub mod pulse_player;
pub mod rodio_player;

use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;

/// Common interface every player implementation must satisfy.
pub trait Player {
    /// Toggle between playing and paused, resuming from the same position.
    fn play_pause(&mut self);
    /// Stop playback immediately.
    fn stop(&mut self);
    /// True once the file has played to the end naturally.
    fn is_finished(&self) -> bool;
    /// True when currently paused (not playing, not stopped).
    fn is_paused(&self) -> bool;
}

/// Entry point called from main.  Dispatches to the correct player type.
pub fn run(player_type: u8, file: &Path) -> Result<()> {
    match player_type {
        1 => run_loop(&mut rodio_player::RodioPlayer::new(file)?),
        2 => run_loop(&mut cpal_player::CpalPlayer::new(file)?),
        3 => run_loop(&mut pulse_player::PulsePlayer::new(file)?),
        _ => anyhow::bail!("--type must be 1, 2, or 3"),
    }
}

/// Enable raw mode, run the spacebar input loop, then restore the terminal.
fn run_loop(player: &mut impl Player) -> Result<()> {
    enable_raw_mode()?;

    // In raw mode \n doesn't CR, so use \r\n explicitly.
    print!("Playing  |  SPACE = pause/resume   Q = quit\r\n");
    print!("[playing]\r\n");
    std::io::stdout().flush()?;

    let result = input_loop(player);

    // Always restore terminal even if the loop errored.
    let _ = disable_raw_mode();
    println!();
    result
}

fn input_loop(player: &mut impl Player) -> Result<()> {
    loop {
        if player.is_finished() {
            print!("\r[done]      \r\n");
            std::io::stdout().flush()?;
            break;
        }

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char(' ') => {
                        player.play_pause();
                        let label = if player.is_paused() { "[paused] " } else { "[playing]" };
                        print!("\r{label}   ");
                        std::io::stdout().flush()?;
                    }
                    KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                        player.stop();
                        print!("\r[stopped]   \r\n");
                        std::io::stdout().flush()?;
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

// ── Shared WAV utilities ──────────────────────────────────────────────────────

const WAV_HEADER_BYTES: usize = 44;
const BYTES_PER_SAMPLE: usize = 3; // 24-bit PCM

/// Read the raw audio bytes from a WAV file (skips the 44-byte header).
pub fn read_wav_bytes(path: &Path) -> Result<Vec<u8>> {
    let raw = std::fs::read(path)?;
    anyhow::ensure!(
        raw.len() > WAV_HEADER_BYTES,
        "file too small to be a valid WAV: {}",
        path.display()
    );
    Ok(raw[WAV_HEADER_BYTES..].to_vec())
}

/// Convert raw 24-bit LE WAV bytes to normalised f32 samples in [-1.0, 1.0].
pub fn wav_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(BYTES_PER_SAMPLE)
        .map(|c| {
            let raw = (c[0] as u32) | ((c[1] as u32) << 8) | ((c[2] as u32) << 16);
            // Sign-extend from bit 23 into a full i32.
            let s24 = (raw << 8) as i32 >> 8;
            s24 as f32 / 8_388_608.0
        })
        .collect()
}

/// Convert raw 24-bit LE WAV bytes to S32LE bytes (4 bytes per sample).
/// Matches the existing ALSA code: each 24-bit sample is left-shifted 8 bits.
pub fn wav_bytes_to_s32le(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity((bytes.len() / BYTES_PER_SAMPLE) * 4);
    for c in bytes.chunks_exact(BYTES_PER_SAMPLE) {
        // S32LE layout (little-endian): [0x00, b0, b1, b2]
        out.push(0x00);  // bits  0-7  (zero — shifted in)
        out.push(c[0]);  // bits  8-15
        out.push(c[1]);  // bits 16-23
        out.push(c[2]);  // bits 24-31 (sign bit preserved)
    }
    out
}

// ── Main-app playback (punch-and-roll) ───────────────────────────────────────
//
// Active backend: PulseAudio (Type 3).
// To swap to a different backend, replace the body of `app_playback_loop`
// below with the desired implementation.  The PlaybackHandle / PlaybackEvent
// interface seen by app.rs stays the same regardless of backend.

/// Events sent from the playback thread to the main thread.
#[derive(Debug)]
pub enum PlaybackEvent {
    /// Playback reached the end of the file naturally.
    Done,
    /// A non-recoverable error occurred during playback.
    Error(String),
}

/// Handle to a running playback thread.
pub struct PlaybackHandle {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    pub rx: mpsc::Receiver<PlaybackEvent>,
}

impl PlaybackHandle {
    /// Signal the thread to stop and wait for it to exit.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }

    /// Signal the thread to stop without blocking.  Used for punch-in so the
    /// main thread stays responsive.  The playback thread sees the flag within
    /// one chunk period (~21 ms) and calls flush itself.
    pub fn signal_stop(self) {
        self.stop.store(true, Ordering::SeqCst);
        // JoinHandle dropped here; thread exits on its own.
    }
}

/// A contiguous region within one WAV file to stream during punch-and-roll.
pub struct PlaybackSegment {
    pub path: PathBuf,
    /// Seconds into the file where playback starts; aligned to a sample boundary.
    pub offset_secs: f64,
    /// How many seconds to play; `f64::INFINITY` means play to EOF.
    pub limit_secs: f64,
}

/// Spawn a playback thread for one or more sequential WAV segments (punch-and-roll).
///
/// Segments are played in order with no gap.  `PlaybackEvent::Done` is sent
/// after the last segment finishes.
pub fn start_playback_segments(segments: Vec<PlaybackSegment>) -> Result<PlaybackHandle> {
    let (tx, rx) = mpsc::sync_channel::<PlaybackEvent>(8);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = Arc::clone(&stop);

    let thread = thread::spawn(move || {
        if let Err(e) = app_playback_loop(&segments, &tx, &stop_clone) {
            let _ = tx.send(PlaybackEvent::Error(e.to_string()));
        }
    });

    Ok(PlaybackHandle { stop, thread: Some(thread), rx })
}

/// Convenience wrapper: play a single WAV file from `offset_secs` to EOF.
pub fn start_playback(path: PathBuf, offset_secs: f64) -> Result<PlaybackHandle> {
    start_playback_segments(vec![PlaybackSegment {
        path,
        offset_secs,
        limit_secs: f64::INFINITY,
    }])
}

// ── PulseAudio streaming loop ─────────────────────────────────────────────────

fn app_playback_loop(
    segments: &[PlaybackSegment],
    tx: &mpsc::SyncSender<PlaybackEvent>,
    stop: &Arc<AtomicBool>,
) -> Result<()> {
    const BYTE_RATE: f64 = 144_000.0; // 48 kHz × 3 bytes/sample
    const CHUNK_SAMPLES: usize = 1024; // ≈21 ms at 48 kHz — same as ALSA period

    let spec = Spec { format: Format::S32le, rate: 48_000, channels: 1 };
    let pa = Simple::new(
        None,                // server  — None = default
        "rustcorder",        // application name
        Direction::Playback,
        None,                // sink — None = default PulseAudio sink
        "punch-and-roll",    // stream name
        &spec,
        None,                // channel map
        None,                // buffer attributes
    )?;

    let mut wav_buf = vec![0u8; CHUNK_SAMPLES * BYTES_PER_SAMPLE];
    let mut s32_buf = vec![0u8; CHUNK_SAMPLES * 4];

    for segment in segments {
        if stop.load(Ordering::Relaxed) {
            let _ = pa.flush();
            return Ok(());
        }

        // Seek to the start of this segment, aligned to a 3-byte sample boundary.
        let raw_offset = (segment.offset_secs * BYTE_RATE) as u64;
        let aligned_offset = raw_offset - (raw_offset % BYTES_PER_SAMPLE as u64);

        // Maximum bytes to read from this segment; u64::MAX = no limit.
        let limit_bytes: u64 = if segment.limit_secs.is_finite() {
            let raw = (segment.limit_secs * BYTE_RATE) as u64;
            raw - (raw % BYTES_PER_SAMPLE as u64)
        } else {
            u64::MAX
        };

        let mut file = File::open(&segment.path)?;
        file.seek(std::io::SeekFrom::Start(WAV_HEADER_BYTES as u64 + aligned_offset))?;

        let mut bytes_played: u64 = 0;

        loop {
            if stop.load(Ordering::Relaxed) {
                let _ = pa.flush();
                return Ok(());
            }

            let remaining = limit_bytes.saturating_sub(bytes_played);
            if remaining == 0 {
                break; // segment limit reached; move to next segment
            }

            // How many bytes to request this chunk, aligned to sample size.
            let chunk_bytes = wav_buf.len().min(remaining as usize);
            let chunk_bytes = chunk_bytes - (chunk_bytes % BYTES_PER_SAMPLE);

            let mut total_read = 0;
            while total_read < chunk_bytes {
                match file.read(&mut wav_buf[total_read..chunk_bytes]) {
                    Ok(0) => break,
                    Ok(n) => total_read += n,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e.into()),
                }
            }

            if total_read == 0 {
                break; // EOF on this segment; move to next
            }

            // Convert 24-bit LE samples to S32LE (left-shift 8) for PulseAudio.
            let samples = total_read / BYTES_PER_SAMPLE;
            for i in 0..samples {
                s32_buf[i * 4]     = 0x00;
                s32_buf[i * 4 + 1] = wav_buf[i * 3];
                s32_buf[i * 4 + 2] = wav_buf[i * 3 + 1];
                s32_buf[i * 4 + 3] = wav_buf[i * 3 + 2];
            }

            pa.write(&s32_buf[..samples * 4])?;
            bytes_played += total_read as u64;
        }
    }

    let _ = pa.drain();
    let _ = tx.send(PlaybackEvent::Done);
    Ok(())
}
