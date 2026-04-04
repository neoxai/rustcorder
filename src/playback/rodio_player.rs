/// Type 1 — rodio
///
/// Uses the `rodio` high-level sink model.  rodio decodes the WAV file via
/// `hound` internally and drives a `cpal` output stream under the hood.
/// Pause/resume is handled by `Sink::pause()` / `Sink::play()`, which freeze
/// and unfreeze the sink at its current position without any manual cursor
/// tracking.
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::Result;
use rodio::{Decoder, OutputStream, Sink};

pub struct RodioPlayer {
    /// The audio sink — controls play/pause/stop.
    sink: Sink,
    /// The output stream must stay alive for the duration of playback;
    /// dropping it silences the device immediately.
    _stream: OutputStream,
    paused: bool,
}

impl RodioPlayer {
    pub fn new(path: &Path) -> Result<Self> {
        let (stream, handle) = OutputStream::try_default()?;
        let sink = Sink::try_new(&handle)?;

        let file = File::open(path)?;
        let source = Decoder::new(BufReader::new(file))?;
        sink.append(source);
        // Start playing immediately; the input loop will handle pause.

        Ok(RodioPlayer { sink, _stream: stream, paused: false })
    }
}

impl super::Player for RodioPlayer {
    fn play_pause(&mut self) {
        if self.paused {
            self.sink.play();
            self.paused = false;
        } else {
            self.sink.pause();
            self.paused = true;
        }
    }

    fn stop(&mut self) {
        self.sink.stop();
    }

    fn is_finished(&self) -> bool {
        self.sink.empty()
    }

    fn is_paused(&self) -> bool {
        self.paused
    }
}
