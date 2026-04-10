use clap::Args;

use anyhow::Result;

/// Configure audio input/output devices
#[derive(Args, Default)]
pub struct ConfigArgs {}

pub fn run(_args: &ConfigArgs) -> Result<()> {
    crate::config::run_config()
}
