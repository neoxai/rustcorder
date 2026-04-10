use std::collections::HashMap;

use anyhow::{bail, Result};

/// Parse a slice of `"key=value"` strings into a map.
///
/// Hard-errors if any key is not in `valid_keys`, or if a string contains no
/// `=` separator.
pub fn parse_options(raw: &[String], valid_keys: &[&str]) -> Result<HashMap<String, String>> {
    let mut map = HashMap::new();
    for item in raw {
        let (key, value) = item
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--options value must be KEY=VALUE, got: {item:?}"))?;
        if !valid_keys.contains(&key) {
            bail!(
                "unknown option key {key:?}. Valid keys for this command: {}",
                valid_keys.join(", ")
            );
        }
        map.insert(key.to_string(), value.to_string());
    }
    Ok(map)
}

// ── Duration helpers ──────────────────────────────────────────────────────────

/// Parse a duration string like `"15s"` or `"15"` as seconds.
pub fn parse_secs(s: &str) -> Option<f64> {
    s.trim().trim_end_matches('s').parse::<f64>().ok()
}

/// Parse a duration string like `"10ms"` or `"10"` as milliseconds.
pub fn parse_ms(s: &str) -> Option<f64> {
    s.trim().trim_end_matches("ms").parse::<f64>().ok()
}

/// Parse a bool string (`"true"` / `"false"`, case-insensitive).
pub fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Some(true),
        "false" | "0" | "no" => Some(false),
        _ => None,
    }
}
