use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};

use alsa::pcm::{Access, Format, HwParams, PCM};
use alsa::Direction;
use anyhow::Result;

// ── Constants ─────────────────────────────────────────────────────────────────

pub const SAMPLE_RATE: u32 = 48_000;
/// Frames per ALSA period.  At 48 kHz this is ~21 ms per batch.
const PERIOD_SIZE: usize = 1024;
/// Channel capacity (number of AudioEvent batches that can be buffered).
const CHANNEL_CAP: usize = 128;

/// Silence threshold in dBFS.  Signals below this are considered silent.
pub const SILENCE_THRESHOLD_DB: f32 = -60.0;
/// Pre-recording validation window in seconds.
pub const PRECHECK_SECONDS: f64 = 2.0;
/// Runtime silence warning threshold in seconds.
pub const SILENCE_WARNING_SECONDS: f64 = 15.0;


// ── Capture types ─────────────────────────────────────────────────────────────

/// Events sent from the capture thread to the main thread.
#[derive(Debug)]
pub enum AudioEvent {
    /// A batch of S24_LE samples and the computed RMS level.
    Samples { data: Vec<i32>, rms_db: f32 },
    /// An ALSA xrun (buffer overrun) occurred — audio may have been lost.
    Xrun,
    /// The ALSA device disappeared (USB unplug).
    DeviceGone,
    /// A non-recoverable ALSA error occurred.
    Error(String),
}

/// Handle to a running capture thread.  Call `stop()` to request shutdown and
/// join the thread.
pub struct CaptureHandle {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    pub rx: mpsc::Receiver<AudioEvent>,
}

impl CaptureHandle {
    /// Signal the thread to stop and wait for it to exit.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Spawn an ALSA capture thread for `alsa_name` (e.g. `"hw:1,0"`).
pub fn start_capture(alsa_name: &str) -> Result<CaptureHandle> {
    let (tx, rx) = mpsc::sync_channel::<AudioEvent>(CHANNEL_CAP);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = Arc::clone(&stop);
    let device = alsa_name.to_string();

    let thread = thread::spawn(move || {
        if let Err(e) = capture_loop(&device, &tx, &stop_clone) {
            let _ = tx.send(AudioEvent::Error(e.to_string()));
        }
    });

    Ok(CaptureHandle { stop, thread: Some(thread), rx })
}


// ── Internal capture loop ─────────────────────────────────────────────────────

fn capture_loop(
    device: &str,
    tx: &mpsc::SyncSender<AudioEvent>,
    stop: &Arc<AtomicBool>,
) -> Result<()> {
    // The Deity VO-7U exposes only S24_3LE (packed 3-byte 24-bit) at the hw:
    // layer.  io_i32() requires S32LE, so we open via plughw: which converts
    // S24_3LE → S32LE in software (lossless bit-shift, no resampling).
    // We then shift the S32LE samples right by 8 to recover the S24 range
    // expected by the WAV writer and RMS calculator.
    let plughw_device: String = device
        .strip_prefix("hw:")
        .map(|rest| format!("plughw:{}", rest))
        .unwrap_or_else(|| device.to_string());

    let pcm = PCM::new(&plughw_device, Direction::Capture, false)?;

    // The Deity presents a stereo interface at the hw: layer even though the
    // capsule is mono.  Try mono first; fall back to stereo and extract the
    // left channel so the WAV file is always mono.
    let actual_channels: u32;
    {
        let hwp = HwParams::any(&pcm)?;
        actual_channels = if hwp.set_channels(1).is_ok() { 1 } else { 2 };
        if actual_channels == 2 {
            hwp.set_channels(2)?;
        }
        hwp.set_rate(SAMPLE_RATE, alsa::ValueOr::Nearest)?;
        hwp.set_format(Format::S32LE)?;
        hwp.set_access(Access::RWInterleaved)?;
        pcm.hw_params(&hwp)?;
    }

    pcm.start()?;
    let io = pcm.io_i32()?;
    // Buffer must hold `PERIOD_SIZE` frames × channels.
    let mut buf = vec![0i32; PERIOD_SIZE * actual_channels as usize];

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }

        match io.readi(&mut buf) {
            Ok(0) => {
                // No frames this period; loop immediately.
            }
            Ok(frames) => {
                // If we opened stereo, extract only the left channel so
                // everything downstream always deals with mono samples.
                // S32LE → S24 range: shift right 8 so the WAV writer and
                // RMS calculator see consistent 24-bit scale.
                let samples: Vec<i32> = if actual_channels == 2 {
                    buf[..frames * 2].iter().step_by(2).map(|&s| s >> 8).collect()
                } else {
                    buf[..frames].iter().map(|&s| s >> 8).collect()
                };
                let rms_db = compute_rms_db(&samples);
                if tx.send(AudioEvent::Samples { data: samples, rms_db }).is_err() {
                    break; // receiver dropped
                }
            }
            Err(e) => {
                // Attempt xrun recovery first.
                match pcm.try_recover(e, false) {
                    Ok(_) => {
                        if tx.send(AudioEvent::Xrun).is_err() {
                            break;
                        }
                        // Re-start capture after xrun recovery.
                        let _ = pcm.start();
                    }
                    Err(e2) => {
                        // Unrecoverable — check if the device was removed.
                        let msg = e2.to_string();
                        if msg.contains("No such device")
                            || msg.contains("Input/output error")
                            || msg.contains("ENODEV")
                        {
                            let _ = tx.send(AudioEvent::DeviceGone);
                        } else {
                            let _ = tx.send(AudioEvent::Error(msg));
                        }
                        break;
                    }
                }
            }
        }
    }

    let _ = pcm.drop();
    Ok(())
}


// ── Signal analysis ───────────────────────────────────────────────────────────

/// Compute the RMS level of a batch of S24_LE samples in dBFS.
///
/// Samples are stored in the lower 24 bits of each i32 (sign-extended), so we
/// sign-extend them to full i32 range before computing.
pub fn compute_rms_db(samples: &[i32]) -> f32 {
    if samples.is_empty() {
        return -100.0;
    }

    let sum_sq: f64 = samples
        .iter()
        .map(|&s| {
            // Sign-extend from bit 23.
            let s24 = (s << 8) >> 8;
            let n = s24 as f64 / 8_388_608.0; // ÷ 2^23
            n * n
        })
        .sum();

    let rms = (sum_sq / samples.len() as f64).sqrt();
    if rms < 1e-10 {
        -100.0
    } else {
        (20.0 * rms.log10()) as f32
    }
}
