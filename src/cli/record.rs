use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Args;

use super::options::{parse_bool, parse_secs, parse_options};

const VALID_KEYS: &[&str] = &[
    "punch_back_time",
    "discard_short_clips",
    "discard_duration",
    "web_mode",
    "browser_open",
    "browser_port",
];

/// Record an audiobook chapter (opens the TUI or web UI)
#[derive(Args, Default)]
pub struct RecordArgs {
    /// Book folder to record into (looked up inside $DEFAULT_RECORDING_DIR).
    /// Skips the setup screen if provided.
    #[arg(long)]
    pub book: Option<String>,

    /// Override a default option (repeatable).
    ///
    /// Valid keys:
    ///   punch_back_time=<Ns>          (default: $PUNCH_BACK_TIME or 15s)
    ///   discard_short_clips=<true|false>  (default: $DISCARD_SHORT_CLIPS or false)
    ///   discard_duration=<Ns>         (default: $DISCARD_DURATION or 1s)
    ///   web_mode=<true|false>         (default: $WEB_MODE or false)
    ///   browser_open=<true|false>     (default: $BROWSER_OPEN or false)
    ///   browser_port=<N>              (default: $BROWSER_PORT or 7474)
    #[arg(long = "options", value_name = "KEY=VALUE")]
    pub options: Vec<String>,
}

/// Fully resolved options for the record command, with all defaults applied.
pub struct RecordOptions {
    pub punch_back_time: f64,
    pub discard_short_clips: bool,
    pub discard_duration_secs: f64,
    pub web_mode: bool,
    pub browser_open: bool,
    pub browser_port: u16,
    /// Base directory where book folders are stored (from $DEFAULT_RECORDING_DIR).
    pub recording_dir: PathBuf,
    /// Book name pre-selected on the command line (skips the setup screen).
    pub book: Option<String>,
}

impl RecordOptions {
    /// Resolve from env vars, then apply any CLI `--options` overrides.
    pub fn resolve(args: &RecordArgs) -> Result<Self> {
        let overrides = parse_options(&args.options, VALID_KEYS)?;

        let punch_back_time = if let Some(v) = overrides.get("punch_back_time") {
            parse_secs(v)
                .ok_or_else(|| anyhow::anyhow!("punch_back_time must be seconds, e.g. 15s"))?
        } else {
            std::env::var("PUNCH_BACK_TIME")
                .ok()
                .and_then(|v| parse_secs(&v))
                .unwrap_or(15.0)
        }
        .max(1.0);

        let discard_short_clips = if let Some(v) = overrides.get("discard_short_clips") {
            parse_bool(v)
                .ok_or_else(|| anyhow::anyhow!("discard_short_clips must be true or false"))?
        } else {
            std::env::var("DISCARD_SHORT_CLIPS")
                .ok()
                .and_then(|v| parse_bool(&v))
                .unwrap_or(false)
        };

        let discard_duration_secs = if let Some(v) = overrides.get("discard_duration") {
            parse_secs(v)
                .ok_or_else(|| anyhow::anyhow!("discard_duration must be seconds, e.g. 1s"))?
        } else {
            std::env::var("DISCARD_DURATION")
                .ok()
                .and_then(|v| parse_secs(&v))
                .unwrap_or(1.0)
        }
        .max(0.0);

        let web_mode = if let Some(v) = overrides.get("web_mode") {
            parse_bool(v).ok_or_else(|| anyhow::anyhow!("web_mode must be true or false"))?
        } else {
            std::env::var("WEB_MODE")
                .ok()
                .and_then(|v| parse_bool(&v))
                .unwrap_or(false)
        };

        let browser_open = if let Some(v) = overrides.get("browser_open") {
            parse_bool(v).ok_or_else(|| anyhow::anyhow!("browser_open must be true or false"))?
        } else {
            std::env::var("BROWSER_OPEN")
                .ok()
                .and_then(|v| parse_bool(&v))
                .unwrap_or(false)
        };

        let browser_port: u16 = if let Some(v) = overrides.get("browser_port") {
            v.trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("browser_port must be a port number, e.g. 7474"))?
        } else {
            std::env::var("BROWSER_PORT")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(7474)
        };

        if browser_port == 0 {
            bail!("browser_port must be > 0");
        }

        let recording_dir = PathBuf::from(
            std::env::var("DEFAULT_RECORDING_DIR").unwrap_or_else(|_| "./recordings".to_string()),
        );

        let book = args.book.clone();

        Ok(RecordOptions {
            punch_back_time,
            discard_short_clips,
            discard_duration_secs,
            web_mode,
            browser_open,
            browser_port,
            recording_dir,
            book,
        })
    }
}
