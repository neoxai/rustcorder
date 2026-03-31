use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

/// Approved USB microphone VID/PID pairs.
/// Only devices listed here may ever be used for recording.
const APPROVED_DEVICES: &[(u16, u16, &str)] = &[
    (0x19f7, 0x003c, "Deity VO-7U"),
];

/// A positively-identified, approved USB microphone with its ALSA hw address.
#[derive(Debug, Clone)]
pub struct MicDevice {
    pub card_index: u32,
    /// ALSA device string, e.g. "hw:1,0"
    pub alsa_name: String,
    /// Human-readable label for display
    pub description: String,
}

/// Scan all ALSA sound cards for exactly one approved USB microphone.
///
/// Returns an error if zero or more than one matching device is found.
pub fn find_approved_mic() -> Result<MicDevice> {
    let mut found: Vec<MicDevice> = Vec::new();

    let dir = match fs::read_dir("/sys/class/sound") {
        Ok(d) => d,
        Err(e) => bail!("Cannot enumerate ALSA devices (/sys/class/sound): {}", e),
    };

    for entry in dir.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // We only care about top-level card entries: card0, card1, …
        let card_num: u32 = match name_str
            .strip_prefix("card")
            .and_then(|s| s.parse().ok())
        {
            Some(n) => n,
            None => continue,
        };

        let card_sysfs = entry.path();

        if let Some((vid, pid, label)) = read_usb_vid_pid(&card_sysfs) {
            if APPROVED_DEVICES.contains(&(vid, pid, label)) {
                if has_capture_pcm(card_num) {
                    let card_id = read_card_id(card_num)
                        .unwrap_or_else(|| format!("card{}", card_num));
                    found.push(MicDevice {
                        card_index: card_num,
                        alsa_name: format!("hw:{},0", card_num),
                        description: format!("{} ({})", label, card_id),
                    });
                }
            }
        }
    }

    match found.len() {
        0 => bail!(
            "No approved USB microphone found.\n\
             Connect the Deity VO-7U and try again."
        ),
        1 => Ok(found.remove(0)),
        n => bail!(
            "{} approved USB microphones detected — connect only one.",
            n
        ),
    }
}

/// Returns true if the microphone's sysfs card directory still exists.
pub fn is_mic_present(mic: &MicDevice) -> bool {
    Path::new(&format!("/sys/class/sound/card{}", mic.card_index)).exists()
}

/// Scan all ALSA sound cards and return a capture-capable device.
///
/// Used only when `--unsafe` is passed.  Prefers the approved Deity mic if
/// it is present; falls back to the first other capture-capable card found.
/// No session-blocking requirement: any mic is accepted.
pub fn find_any_capture_device() -> Result<MicDevice> {
    // Try the approved mic first — preferred even in unsafe mode.
    if let Ok(approved) = find_approved_mic() {
        return Ok(approved);
    }

    let dir = match std::fs::read_dir("/sys/class/sound") {
        Ok(d) => d,
        Err(e) => bail!("Cannot enumerate ALSA devices (/sys/class/sound): {}", e),
    };

    let mut cards: Vec<u32> = dir
        .flatten()
        .filter_map(|e| {
            let n = e.file_name();
            let s = n.to_string_lossy();
            s.strip_prefix("card")
                .and_then(|rest| rest.parse::<u32>().ok())
        })
        .collect();

    cards.sort_unstable();

    for card_num in cards {
        if has_capture_pcm(card_num) {
            let card_id = read_card_id(card_num)
                .unwrap_or_else(|| format!("card{}", card_num));
            return Ok(MicDevice {
                card_index: card_num,
                alsa_name: format!("hw:{},0", card_num),
                description: format!("{} (card{})", card_id, card_num),
            });
        }
    }

    bail!("No capture-capable ALSA device found.")
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Walk up the sysfs tree from the card directory until we find idVendor /
/// idProduct files.  Returns (vid, pid, matching_label) or None.
fn read_usb_vid_pid(card_sysfs: &Path) -> Option<(u16, u16, &'static str)> {
    // card_sysfs is a symlink; canonicalise to resolve it.
    let real = fs::canonicalize(card_sysfs).ok()?;

    let mut search = real.as_path();

    // Walk upward up to ~10 levels to reach the USB device directory.
    for _ in 0..10 {
        let vid_path = search.join("idVendor");
        let pid_path = search.join("idProduct");

        if vid_path.exists() && pid_path.exists() {
            let vid_str = fs::read_to_string(&vid_path).ok()?;
            let pid_str = fs::read_to_string(&pid_path).ok()?;

            let vid = u16::from_str_radix(vid_str.trim(), 16).ok()?;
            let pid = u16::from_str_radix(pid_str.trim(), 16).ok()?;

            // Match against approved table and return the label.
            for &(avid, apid, label) in APPROVED_DEVICES {
                if avid == vid && apid == pid {
                    return Some((vid, pid, label));
                }
            }
            // VID/PID found but not approved — stop walking.
            return None;
        }

        search = search.parent()?;
    }

    None
}

/// Returns true if there is at least one capture PCM node for this card in
/// /sys/class/sound (file name pattern: pcmC<N>D<M>c).
fn has_capture_pcm(card_num: u32) -> bool {
    let prefix = format!("pcmC{}D", card_num);
    fs::read_dir("/sys/class/sound")
        .ok()
        .map(|dir| {
            dir.flatten().any(|e| {
                let s = e.file_name();
                let s = s.to_string_lossy();
                s.starts_with(&prefix) && s.ends_with('c')
            })
        })
        .unwrap_or(false)
}

/// Read the short identifier for a card from /proc/asound/card<N>/id.
fn read_card_id(card_num: u32) -> Option<String> {
    let path = PathBuf::from(format!("/proc/asound/card{}/id", card_num));
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}
