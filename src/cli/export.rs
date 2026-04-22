use anyhow::{bail, Result};
use clap::Args;

use crate::export::timeline::{export_chapter, ExportOptions};

use super::options::{parse_bool, parse_ms, parse_options, parse_secs};

const VALID_KEYS: &[&str] = &["crossfade_time", "discard_short_clips", "discard_duration"];

/// Export a recorded chapter to a final mixed WAV file
#[derive(Args)]
pub struct ExportArgs {
    /// Book directory name (must contain Chapter_NN_timeline.txt)
    #[arg(long)]
    pub book: String,

    /// Chapter number to export
    #[arg(long)]
    pub chapter: u32,

    /// Override a default option (repeatable).
    ///
    /// Valid keys:
    ///   crossfade_time=<Nms>          (default: $CROSSFADE_TIME or 10ms)
    ///   discard_short_clips=<true|false>  (default: $DISCARD_SHORT_CLIPS or false)
    ///   discard_duration=<Ns>         (default: $DISCARD_DURATION or 1s)
    #[arg(long = "options", value_name = "KEY=VALUE")]
    pub options: Vec<String>,
}

pub fn run(args: &ExportArgs) -> Result<()> {
    let overrides = parse_options(&args.options, VALID_KEYS)?;

    // Resolve each value: CLI override → env var → hard default.
    let crossfade_ms = if let Some(v) = overrides.get("crossfade_time") {
        parse_ms(v).ok_or_else(|| anyhow::anyhow!("crossfade_time must be a number of ms, e.g. 10ms"))?
    } else {
        std::env::var("CROSSFADE_TIME")
            .ok()
            .and_then(|v| parse_ms(&v))
            .unwrap_or(10.0)
    };

    let discard_short_clips = if let Some(v) = overrides.get("discard_short_clips") {
        parse_bool(v).ok_or_else(|| anyhow::anyhow!("discard_short_clips must be true or false"))?
    } else {
        std::env::var("DISCARD_SHORT_CLIPS")
            .ok()
            .and_then(|v| parse_bool(&v))
            .unwrap_or(false)
    };

    let discard_duration_secs = if let Some(v) = overrides.get("discard_duration") {
        parse_secs(v).ok_or_else(|| anyhow::anyhow!("discard_duration must be a number of seconds, e.g. 1s"))?
    } else {
        std::env::var("DISCARD_DURATION")
            .ok()
            .and_then(|v| parse_secs(&v))
            .unwrap_or(1.0)
    };

    if crossfade_ms < 0.0 {
        bail!("crossfade_time must be >= 0");
    }
    if discard_duration_secs < 0.0 {
        bail!("discard_duration must be >= 0");
    }

    let opts = ExportOptions {
        crossfade_ms,
        discard_short_clips,
        discard_duration_secs: discard_duration_secs.max(0.0),
    };

    let recording_dir = std::path::PathBuf::from(
        std::env::var("DEFAULT_RECORDING_DIR").unwrap_or_else(|_| "./recordings".to_string()),
    );
    let book_dir = recording_dir.join(&args.book);
    let out = export_chapter(&book_dir, args.chapter, &opts)?;
    println!("Exported: {}", out.display());
    Ok(())
}
