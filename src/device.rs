use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

/// Approved USB microphone VID/PID pairs.
/// Only devices listed here may ever be used for recording.
const APPROVED_DEVICES: &[(u16, u16, &str)] = &[
    (0x19f7, 0x003c, "Deity VO-7U"),
];

/// File written by `--config` wizard and read back in `--unsafe` mode.
const CHOSEN_DEVICES_FILE: &str = "chosen.devices.txt";

// ── Public types ──────────────────────────────────────────────────────────────

/// A positively-identified, approved USB microphone with its ALSA hw address.
#[derive(Debug, Clone)]
pub struct MicDevice {
    pub card_index: u32,
    /// ALSA device string, e.g. "hw:1,0"
    pub alsa_name: String,
    /// Human-readable label for display
    pub description: String,
}

/// A generic ALSA device (capture or playback) enriched with metadata for the
/// `--config` picker.  Only `alsa_name` and `description` are persisted.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub alsa_name: String,
    /// Short product / card name (persisted to chosen.devices.txt).
    pub description: String,
    /// Bus / driver type: "USB", "HDA", "Bluetooth", "Platform", "Virtual", …
    pub bus_type: String,
    /// Optional second line shown in the picker (manufacturer, serial, detail).
    pub detail: Option<String>,
}

/// A pair of pre-selected capture and playback devices written by `--config`.
pub struct ChosenDevices {
    pub capture: DeviceInfo,
    pub playback: DeviceInfo,
}

// ── Approved-mic detection (normal mode) ─────────────────────────────────────

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
pub fn find_any_capture_device() -> Result<MicDevice> {
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

// ── Device enumeration (--config wizard) ─────────────────────────────────────

/// Return all ALSA cards that expose at least one capture PCM node, sorted by
/// card index and annotated with bus type and extra metadata.
pub fn enumerate_capture_devices() -> Vec<DeviceInfo> {
    sorted_card_indices()
        .into_iter()
        .filter(|&n| has_capture_pcm(n))
        .map(|n| build_device_info(n))
        .collect()
}

/// Return all ALSA cards that expose at least one playback PCM node, prefixed
/// by the ALSA `"default"` virtual device (covers PipeWire / PulseAudio,
/// including Bluetooth headphones managed by the system audio daemon).
pub fn enumerate_playback_devices() -> Vec<DeviceInfo> {
    let mut result = vec![DeviceInfo {
        alsa_name: "default".to_string(),
        description: "System default".to_string(),
        bus_type: "Virtual".to_string(),
        detail: Some(
            "PipeWire / PulseAudio mix — includes Bluetooth headphones".to_string(),
        ),
    }];

    let hw_devices: Vec<DeviceInfo> = sorted_card_indices()
        .into_iter()
        .filter(|&n| has_playback_pcm(n))
        .map(|n| build_device_info(n))
        .collect();

    result.extend(hw_devices);
    result
}

// ── chosen.devices.txt persistence ───────────────────────────────────────────

/// Load the devices previously selected by the `--config` wizard.
/// Returns `None` if the file is absent or malformed.
pub fn load_chosen_devices() -> Option<ChosenDevices> {
    let content = fs::read_to_string(CHOSEN_DEVICES_FILE).ok()?;
    let mut capture_device = None;
    let mut capture_name: Option<String> = None;
    let mut playback_device = None;
    let mut playback_name: Option<String> = None;

    for line in content.lines() {
        if let Some(v) = line.strip_prefix("CAPTURE_DEVICE=") {
            capture_device = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("CAPTURE_NAME=") {
            capture_name = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("PLAYBACK_DEVICE=") {
            playback_device = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("PLAYBACK_NAME=") {
            playback_name = Some(v.trim().to_string());
        }
    }

    Some(ChosenDevices {
        capture: DeviceInfo {
            alsa_name: capture_device?,
            description: capture_name.unwrap_or_default(),
            bus_type: String::new(),
            detail: None,
        },
        playback: DeviceInfo {
            alsa_name: playback_device?,
            description: playback_name.unwrap_or_default(),
            bus_type: String::new(),
            detail: None,
        },
    })
}

/// Write the selected devices to `chosen.devices.txt`.
pub fn save_chosen_devices(chosen: &ChosenDevices) -> Result<()> {
    let mut f = fs::File::create(CHOSEN_DEVICES_FILE)?;
    writeln!(f, "CAPTURE_DEVICE={}", chosen.capture.alsa_name)?;
    writeln!(f, "CAPTURE_NAME={}", chosen.capture.description)?;
    writeln!(f, "PLAYBACK_DEVICE={}", chosen.playback.alsa_name)?;
    writeln!(f, "PLAYBACK_NAME={}", chosen.playback.description)?;
    Ok(())
}

// ── Internal: card metadata assembly ─────────────────────────────────────────

/// Build a fully-annotated `DeviceInfo` for one ALSA card index.
fn build_device_info(card_num: u32) -> DeviceInfo {
    let alsa_name = format!("hw:{},0", card_num);

    // Primary source: /proc/asound/cards — gives driver type and a detail line.
    let proc_info = read_proc_card_info(card_num);

    // Derive bus_type from the ALSA driver string, then from sysfs path.
    let sysfs = PathBuf::from(format!("/sys/class/sound/card{}", card_num));
    let bus_type = derive_bus_type(proc_info.as_ref().map(|i| i.driver.as_str()), &sysfs);

    // Product name: for USB prefer sysfs `product` file; otherwise use the
    // name from /proc/asound/cards; fall back to the short card id.
    let description = if bus_type == "USB" {
        read_sysfs_string(&sysfs, "product")
            .or_else(|| proc_info.as_ref().map(|i| i.name.clone()))
            .or_else(|| read_card_id(card_num))
            .unwrap_or_else(|| format!("card{}", card_num))
    } else {
        proc_info
            .as_ref()
            .map(|i| i.name.clone())
            .or_else(|| read_card_id(card_num))
            .unwrap_or_else(|| format!("card{}", card_num))
    };

    // Detail line: for USB, show manufacturer + serial; otherwise the
    // hardware detail string from /proc/asound/cards.
    let detail: Option<String> = if bus_type == "USB" {
        let mfr = read_sysfs_string(&sysfs, "manufacturer");
        let ser = read_sysfs_string(&sysfs, "serial");
        match (mfr, ser) {
            (Some(m), Some(s)) => Some(format!("{}  ·  S/N: {}", m, s)),
            (Some(m), None) => Some(m),
            (None, Some(s)) => Some(format!("S/N: {}", s)),
            (None, None) => proc_info.as_ref().map(|i| i.detail.clone()),
        }
    } else {
        proc_info.as_ref().map(|i| i.detail.clone())
    };

    DeviceInfo { alsa_name, description, bus_type, detail }
}

// ── Internal: /proc/asound/cards parser ──────────────────────────────────────

struct ProcCardInfo {
    /// Driver/type string, e.g. "HDA-Intel", "USB-Audio", "acp-pdm-mach".
    driver: String,
    /// Short product name after the " - " separator on line 1.
    name: String,
    /// Continuation detail line (trimmed).
    detail: String,
}

/// Parse one card's entry from `/proc/asound/cards`.
///
/// Format (two lines per card):
/// ```text
///  N [SHORT_ID       ]: DRIVER - NAME
///                       DETAIL
/// ```
fn read_proc_card_info(card_num: u32) -> Option<ProcCardInfo> {
    let content = fs::read_to_string("/proc/asound/cards").ok()?;
    let num_str = card_num.to_string();
    let mut lines = content.lines();

    while let Some(line) = lines.next() {
        // Each card occupies two lines. Line 1 looks like:
        //   " N [SHORT_ID       ]: DRIVER - NAME"
        // We match by stripping leading whitespace and checking for exactly
        // the card number followed by " [".
        let trimmed = line.trim_start();
        let after_num = match trimmed.strip_prefix(num_str.as_str()) {
            Some(s) => s,
            None => continue, // number doesn't match — keep looping
        };
        if !after_num.starts_with(" [") {
            continue; // false match (e.g. "10" when looking for "1")
        }

        // Extract ]: DRIVER - NAME
        let bracket_close = match line.find("]:") {
            Some(i) => i,
            None => continue,
        };
        let rest = line[bracket_close + 2..].trim();

        let (driver, name) = if let Some(dash) = rest.find(" - ") {
            (rest[..dash].trim().to_string(), rest[dash + 3..].trim().to_string())
        } else {
            (rest.to_string(), rest.to_string())
        };

        let detail = lines
            .next()
            .map(|l| l.trim().to_string())
            .unwrap_or_default();

        return Some(ProcCardInfo { driver, name, detail });
    }
    None
}

// ── Internal: bus-type detection ─────────────────────────────────────────────

/// Determine a human-readable bus type from the ALSA driver string and/or the
/// sysfs device path.
fn derive_bus_type(driver: Option<&str>, card_sysfs: &Path) -> String {
    // The ALSA driver string is the most reliable indicator.
    if let Some(d) = driver {
        let dl = d.to_lowercase();
        if dl == "usb-audio" || dl.starts_with("usb") {
            return "USB".to_string();
        }
        if dl == "hda-intel" || dl.starts_with("hda") {
            return "HDA".to_string();
        }
        if dl.contains("bluetooth") || dl.contains("btaudio") {
            return "Bluetooth".to_string();
        }
        // AMD Audio Co-Processor (laptop internal mic/speaker)
        if dl.starts_with("acp") || dl.contains("pdm") {
            return "Platform".to_string();
        }
    }

    // Fall back: inspect the sysfs symlink target for structural clues.
    if let Ok(real) = fs::canonicalize(card_sysfs) {
        let s = real.to_string_lossy();
        if s.contains("/usb") {
            return "USB".to_string();
        }
        if s.contains("bluetooth") {
            return "Bluetooth".to_string();
        }
        if s.contains("hdaudio") {
            return "HDA".to_string();
        }
        if s.contains("platform") {
            return "Platform".to_string();
        }
        if s.contains("/pci") {
            return "PCI".to_string();
        }
    }

    "Unknown".to_string()
}

// ── Internal: sysfs string helpers ───────────────────────────────────────────

/// Walk up from `card_sysfs` (resolving the symlink first) up to 10 levels
/// looking for a sysfs file named `attr`.  Returns the trimmed content or None.
fn read_sysfs_string(card_sysfs: &Path, attr: &str) -> Option<String> {
    let real = fs::canonicalize(card_sysfs).ok()?;
    let mut search: &Path = &real;
    for _ in 0..10 {
        let p = search.join(attr);
        if let Ok(s) = fs::read_to_string(&p) {
            let s = s.trim().to_string();
            if !s.is_empty() {
                return Some(s);
            }
        }
        search = search.parent()?;
    }
    None
}

// ── Internal: PCM-node and card-id helpers ────────────────────────────────────

fn sorted_card_indices() -> Vec<u32> {
    let dir = match fs::read_dir("/sys/class/sound") {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let mut cards: Vec<u32> = dir
        .flatten()
        .filter_map(|e| {
            let n = e.file_name();
            let s = n.to_string_lossy();
            s.strip_prefix("card")
                .and_then(|r| r.parse::<u32>().ok())
        })
        .collect();
    cards.sort_unstable();
    cards
}

/// Returns true if there is at least one capture PCM node for this card
/// (pattern: pcmC<N>D<M>c).
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

/// Returns true if there is at least one playback PCM node for this card
/// (pattern: pcmC<N>D<M>p).
fn has_playback_pcm(card_num: u32) -> bool {
    let prefix = format!("pcmC{}D", card_num);
    fs::read_dir("/sys/class/sound")
        .ok()
        .map(|dir| {
            dir.flatten().any(|e| {
                let s = e.file_name();
                let s = s.to_string_lossy();
                s.starts_with(&prefix) && s.ends_with('p')
            })
        })
        .unwrap_or(false)
}

/// Read the short identifier for a card from /proc/asound/card<N>/id.
fn read_card_id(card_num: u32) -> Option<String> {
    let path = PathBuf::from(format!("/proc/asound/card{}/id", card_num));
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

// ── Internal: approved-mic USB VID/PID walker ─────────────────────────────────

/// Walk up the sysfs tree from the card directory until we find idVendor /
/// idProduct files.  Returns (vid, pid, matching_label) or None.
fn read_usb_vid_pid(card_sysfs: &Path) -> Option<(u16, u16, &'static str)> {
    let real = fs::canonicalize(card_sysfs).ok()?;
    let mut search = real.as_path();

    for _ in 0..10 {
        let vid_path = search.join("idVendor");
        let pid_path = search.join("idProduct");

        if vid_path.exists() && pid_path.exists() {
            let vid_str = fs::read_to_string(&vid_path).ok()?;
            let pid_str = fs::read_to_string(&pid_path).ok()?;

            let vid = u16::from_str_radix(vid_str.trim(), 16).ok()?;
            let pid = u16::from_str_radix(pid_str.trim(), 16).ok()?;

            for &(avid, apid, label) in APPROVED_DEVICES {
                if avid == vid && apid == pid {
                    return Some((vid, pid, label));
                }
            }
            return None;
        }

        search = search.parent()?;
    }

    None
}
