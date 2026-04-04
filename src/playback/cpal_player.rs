/// Type 2 — cpal (callback model)
///
/// The entire WAV file is pre-loaded into memory as f32 samples.  A cpal
/// output stream is opened once and kept alive; its data callback advances a
/// shared cursor through the buffer.  Pausing simply stops advancing the
/// cursor — the callback writes silence instead — so resume picks up at
/// exactly the right sample without reopening any stream.
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};

struct State {
    samples: Vec<f32>,
    cursor: usize,
    paused: bool,
    finished: bool,
}

pub struct CpalPlayer {
    state: Arc<Mutex<State>>,
    /// Stream must be kept alive; dropping it stops playback.
    _stream: Stream,
}

impl CpalPlayer {
    pub fn new(path: &Path) -> Result<Self> {
        let wav_bytes = super::read_wav_bytes(path)?;
        let samples = super::wav_bytes_to_f32(&wav_bytes);

        let state = Arc::new(Mutex::new(State {
            samples,
            cursor: 0,
            paused: false,
            finished: false,
        }));

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("no default output device found"))?;

        // Use the device's preferred channel count but request 48 kHz to
        // match the WAV file.  No resampling is performed — if the hardware
        // doesn't support 48 kHz, cpal will return an error here.
        let default_cfg = device.default_output_config()?;
        let channels = default_cfg.channels() as usize;
        let config = StreamConfig {
            channels: default_cfg.channels(),
            sample_rate: cpal::SampleRate(48_000),
            buffer_size: cpal::BufferSize::Default,
        };

        let state_cb = Arc::clone(&state);
        let stream = device.build_output_stream(
            &config,
            move |output: &mut [f32], _| {
                let mut st = state_cb.lock().unwrap();
                // Each frame has `channels` samples; write the same mono
                // sample to every channel (duplicate for stereo etc.).
                for frame in output.chunks_mut(channels) {
                    let sample = if st.paused || st.cursor >= st.samples.len() {
                        if !st.paused && st.cursor >= st.samples.len() {
                            st.finished = true;
                        }
                        0.0_f32
                    } else {
                        let s = st.samples[st.cursor];
                        st.cursor += 1;
                        s
                    };
                    for ch in frame.iter_mut() {
                        *ch = sample;
                    }
                }
            },
            |err| eprintln!("cpal stream error: {err}"),
            None,
        )?;

        stream.play()?;

        Ok(CpalPlayer { state, _stream: stream })
    }
}

impl super::Player for CpalPlayer {
    fn play_pause(&mut self) {
        let mut st = self.state.lock().unwrap();
        st.paused = !st.paused;
    }

    fn stop(&mut self) {
        let mut st = self.state.lock().unwrap();
        // Jump cursor to end so is_finished() returns true.
        st.cursor = st.samples.len();
        st.finished = true;
    }

    fn is_finished(&self) -> bool {
        self.state.lock().unwrap().finished
    }

    fn is_paused(&self) -> bool {
        self.state.lock().unwrap().paused
    }
}
