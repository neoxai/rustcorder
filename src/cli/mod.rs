use clap::{Parser, Subcommand};

pub mod config;
pub mod export;
pub mod options;
pub mod playback;
pub mod record;

/// Audiobook recording studio
#[derive(Parser)]
#[command(name = "rustcorder", version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Record an audiobook (default when no command is given)
    Record(record::RecordArgs),
    /// Export a chapter to a final WAV file
    Export(export::ExportArgs),
    /// Configure audio devices
    Config(config::ConfigArgs),
    /// Play back a WAV file
    Playback(playback::PlaybackArgs),
}
