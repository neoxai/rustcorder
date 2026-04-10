use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, ValueEnum};

/// Play back a WAV file
#[derive(Args)]
pub struct PlaybackArgs {
    /// WAV file to play
    #[arg(long)]
    pub file: PathBuf,

    /// Audio playback backend [default: rodio]
    #[arg(long, default_value = "rodio")]
    pub backend: Backend,
}

/// Playback backend selection
#[derive(Clone, ValueEnum)]
pub enum Backend {
    /// Rodio (default) — cross-platform, no system daemon required
    Rodio,
    /// CPAL — low-level cross-platform audio I/O
    Cpal,
    /// PulseAudio — Linux PulseAudio daemon
    Pulse,
}

impl Backend {
    fn as_player_type(&self) -> u8 {
        match self {
            Backend::Rodio => 1,
            Backend::Cpal  => 2,
            Backend::Pulse => 3,
        }
    }
}

pub fn run(args: &PlaybackArgs) -> Result<()> {
    crate::playback::run(args.backend.as_player_type(), &args.file)
}
