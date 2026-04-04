/// Type 3 — PulseAudio via libpulse-simple-binding
///
/// The WAV file is pre-loaded and converted to S32LE (same expansion used by
/// the existing ALSA path).  A background thread feeds chunks to a
/// `pa_simple` stream.  On pause the thread sets a flag and calls
/// `Simple::flush()` to drain PulseAudio's internal buffer so audio stops
/// promptly; on resume it continues writing from the saved cursor position.
///
/// Build requirement: libpulse-dev must be installed.
///   sudo apt install libpulse-dev
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::Result;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;

struct State {
    cursor: usize,
    paused: bool,
    stop: bool,
    needs_flush: bool,
    finished: bool,
}

pub struct PulsePlayer {
    state: Arc<Mutex<State>>,
    thread: Option<JoinHandle<()>>,
}

impl PulsePlayer {
    pub fn new(path: &Path) -> Result<Self> {
        let wav_bytes = super::read_wav_bytes(path)?;
        let samples_s32 = super::wav_bytes_to_s32le(&wav_bytes);
        let samples = Arc::new(samples_s32);

        let state = Arc::new(Mutex::new(State {
            cursor: 0,
            paused: false,
            stop: false,
            needs_flush: false,
            finished: false,
        }));

        let samples_t = Arc::clone(&samples);
        let state_t = Arc::clone(&state);

        let thread = thread::spawn(move || {
            write_thread(samples_t, state_t);
        });

        Ok(PulsePlayer { state, thread: Some(thread) })
    }
}

impl super::Player for PulsePlayer {
    fn play_pause(&mut self) {
        let mut st = self.state.lock().unwrap();
        if st.paused {
            st.paused = false;
        } else {
            st.paused = true;
            // Ask the write thread to flush PulseAudio's buffer so audio
            // stops within one chunk (~21 ms) rather than after the whole
            // PA buffer drains.
            st.needs_flush = true;
        }
    }

    fn stop(&mut self) {
        self.state.lock().unwrap().stop = true;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }

    fn is_finished(&self) -> bool {
        self.state.lock().unwrap().finished
    }

    fn is_paused(&self) -> bool {
        self.state.lock().unwrap().paused
    }
}

impl Drop for PulsePlayer {
    fn drop(&mut self) {
        self.state.lock().unwrap().stop = true;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

// ── Background write thread ───────────────────────────────────────────────────

fn write_thread(samples: Arc<Vec<u8>>, state: Arc<Mutex<State>>) {
    let spec = Spec {
        format: Format::S32le,
        rate: 48_000,
        channels: 1,
    };

    let pa = match Simple::new(
        None,                // server  — None = default
        "rustcorder",        // application name
        Direction::Playback,
        None,                // sink device — None = default
        "playback",          // stream name
        &spec,
        None,                // channel map
        None,                // buffer attributes
    ) {
        Ok(pa) => pa,
        Err(e) => {
            eprintln!("PulseAudio connect failed: {e}");
            state.lock().unwrap().finished = true;
            return;
        }
    };

    // 4096 bytes = 1024 S32LE samples ≈ 21 ms at 48 kHz — same granularity
    // as the existing ALSA period size.
    const CHUNK: usize = 4096;

    loop {
        let (stop, paused, needs_flush, cursor) = {
            let st = state.lock().unwrap();
            (st.stop, st.paused, st.needs_flush, st.cursor)
        };

        if stop {
            break;
        }

        // Handle flush request before checking paused so the flush happens
        // as soon as the pause is requested.
        if needs_flush {
            let _ = pa.flush();
            state.lock().unwrap().needs_flush = false;
            continue;
        }

        if paused {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        if cursor >= samples.len() {
            let _ = pa.drain();
            state.lock().unwrap().finished = true;
            break;
        }

        let end = (cursor + CHUNK).min(samples.len());
        match pa.write(&samples[cursor..end]) {
            Ok(_) => {
                state.lock().unwrap().cursor = end;
            }
            Err(e) => {
                eprintln!("PulseAudio write error: {e}");
                state.lock().unwrap().finished = true;
                break;
            }
        }
    }
}
