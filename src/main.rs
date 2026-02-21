use anyhow::{bail, Context, Result};
use hound::{WavSpec, WavWriter};
use std::fs;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdout, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};
use walkdir::WalkDir;

const VID: &str = "19f7";
const PID: &str = "003c";
const SESSION_FILE: &str = "session.save.txt";
const SAMPLE_RATE: u32 = 48000;
const CHANNELS: u16 = 1;
const BITS_PER_SAMPLE: u16 = 24;
const PRE_RECORD_SECONDS: u64 = 2;
const SILENCE_THRESHOLD_DBFS: f32 = -60.0;
const RUNTIME_SILENCE_SECONDS: u64 = 15;

#[derive(Debug, serde::Deserialize, serde::Serialize, Default)]
struct Session {
    book: String,
    chapter: u32,
    part: u32,
}

fn load_session() -> Session {
    if let Ok(s) = fs::read_to_string(SESSION_FILE) {
        let mut sess = Session::default();
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("BOOK=") {
                sess.book = rest.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("CHAPTER=") {
                sess.chapter = rest.trim().parse().unwrap_or(1);
            } else if let Some(rest) = line.strip_prefix("PART=") {
                sess.part = rest.trim().parse().unwrap_or(1);
            }
        }
        if sess.book.is_empty() {
            sess.book = "Untitled".into();
        }
        if sess.chapter == 0 {
            sess.chapter = 1;
        }
        if sess.part == 0 {
            sess.part = 1;
        }
        sess
    } else {
        Session {
            book: String::new(),
            chapter: 1,
            part: 1,
        }
    }
}

fn save_session(sess: &Session) -> Result<()> {
    let content = format!("BOOK={}\nCHAPTER={:02}\nPART={}\n", sess.book, sess.chapter, sess.part);
    fs::write(SESSION_FILE, content).context("writing session file")?;
    Ok(())
}

fn choose_or_edit_session(mut sess: Session) -> Result<Session> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    if sess.book.is_empty() {
        print!("Enter Book name: ");
        stdout.flush()?;
        stdin.read_line(&mut sess.book)?;
        sess.book = sess.book.trim().to_string();
    } else {
        println!("Last session: Book='{}' Chapter={:02} Part={}", sess.book, sess.chapter, sess.part);
        print!("Press Enter to accept or type new Book name: ");
        stdout.flush()?;
        let mut input = String::new();
        stdin.read_line(&mut input)?;
        if !input.trim().is_empty() {
            sess.book = input.trim().to_string();
        }
    }

    print!("Chapter number (two digits) [{:02}]: ", sess.chapter);
    stdout.flush()?;
    let mut input = String::new();
    stdin.read_line(&mut input)?;
    if let Ok(n) = input.trim().parse::<u32>() {
        if n > 0 {
            sess.chapter = n;
            sess.part = 1;
        }
    }
    Ok(sess)
}

fn detect_matching_cards() -> Result<Vec<u32>> {
    let mut matches = Vec::new();
    for entry in fs::read_dir("/sys/class/sound").context("reading /sys/class/sound")? {
        let e = entry?;
        let file_name = e.file_name();
        let name = file_name.to_string_lossy();
        if name.starts_with("card") {
            let card_path = e.path();
            let device_link = card_path.join("device");
            if !device_link.exists() {
                continue;
            }
            let mut cur = device_link.clone();
            let mut found = false;
            for _ in 0..8 {
                if cur.join("idVendor").exists() && cur.join("idProduct").exists() {
                    let vid = fs::read_to_string(cur.join("idVendor"))?.trim().to_lowercase();
                    let pid = fs::read_to_string(cur.join("idProduct"))?.trim().to_lowercase();
                    if vid == VID && pid == PID {
                        if let Some(s) = name.strip_prefix("card") {
                            if let Ok(idx) = s.parse::<u32>() {
                                matches.push(idx);
                            }
                        }
                    }
                    found = true;
                    break;
                }
                if let Some(parent) = cur.parent() {
                    cur = parent.to_path_buf();
                } else {
                    break;
                }
            }
            if !found {
            }
        }
    }
    Ok(matches)
}

fn card_to_alsa_device(card: u32) -> String {
    format!("hw:{}", card)
}

fn build_output_path(book: &str, chapter: u32, part: u32) -> Result<PathBuf> {
    let dir = Path::new(book);
    if !dir.exists() {
        fs::create_dir_all(dir).context("creating book directory")?;
    }
    let file_name = format!("Chapter_{:02}_part{:03}.wav", chapter, part);
    Ok(dir.join(file_name))
}

fn find_next_part(book: &str, chapter: u32) -> u32 {
    let dir = Path::new(book);
    if !dir.exists() {
        return 1;
    }
    let mut max = 0u32;
    for entry in WalkDir::new(dir).max_depth(1) {
        if let Ok(ent) = entry {
            if ent.file_type().is_file() {
                if let Some(name) = ent.path().file_name().and_then(|s| s.to_str()) {
                    if name.starts_with(&format!("Chapter_{:02}_part", chapter)) && name.ends_with(".wav") {
                        let part_str = name.trim_end_matches(".wav").rsplit("part").next().unwrap_or("");
                        if let Ok(idx) = part_str.parse::<u32>() {
                            if idx > max {
                                max = idx;
                            }
                        }
                    }
                }
            }
        }
    }
    max + 1
}

fn compute_rms_dbfs(samples: &[i32]) -> f32 {
    if samples.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut sum_sq = 0f64;
    for &s in samples {
        let f = s as f64 / (1i64 << 23) as f64;
        sum_sq += f * f;
    }
    let mean_sq = sum_sq / (samples.len() as f64);
    let rms = mean_sq.sqrt();
    20.0 * (rms.max(1e-12)).log10()
}

fn spawn_arecord(device: &str) -> Result<(std::process::Child, ChildStdout)> {
    let mut cmd = Command::new("arecord");
    cmd.arg("-D")
        .arg(device)
        .arg("-f")
        .arg("S24_LE")
        .arg("-r")
        .arg(format!("{}", SAMPLE_RATE))
        .arg("-c")
        .arg(format!("{}", CHANNELS))
        .arg("-t")
        .arg("raw")
        .arg("-")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = cmd.spawn().context("spawning arecord")?;
    let stdout = child
        .stdout
        .take()
        .context("failed to capture arecord stdout")?;
    Ok((child, stdout))
}

fn read_samples_from_raw(reader: &mut dyn Read, buf_samples: &mut Vec<i32>, max_samples: usize) -> io::Result<usize> {
    buf_samples.clear();
    let mut bytes = vec![0u8; max_samples * 3];
    let mut read = 0usize;
    while read < bytes.len() {
        match reader.read(&mut bytes[read..]) {
            Ok(0) => break,
            Ok(n) => read += n,
            Err(e) => return Err(e),
        }
    }
    let samples = read / 3;
    for i in 0..samples {
        let b0 = bytes[i * 3] as u32;
        let b1 = bytes[i * 3 + 1] as u32;
        let b2 = bytes[i * 3 + 2] as u32;
        let val = (b2 << 16) | (b1 << 8) | b0;
        let signed = if (val & 0x800000) != 0 {
            (val | 0xff000000) as i32
        } else {
            val as i32
        };
        buf_samples.push(signed);
    }
    Ok(samples)
}

fn write_wav_stream(
    mut stdout_reader: ChildStdout,
    out_path: PathBuf,
    stop_flag: Arc<AtomicBool>,
    runtime_silence_warn: Arc<AtomicBool>,
) -> Result<()> {
    let spec = WavSpec {
        channels: CHANNELS,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: BITS_PER_SAMPLE,
        sample_format: hound::SampleFormat::Int,
    };
    let file = File::create(&out_path).context("creating wav file")?;
    let mut writer = WavWriter::new(file, spec).context("creating wav writer")?;

    let mut buf: Vec<i32> = Vec::with_capacity(4096);
    let mut last_non_silent = Instant::now();
    let mut spinner = vec!["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    let mut spin_i = 0usize;

    loop {
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }
        let n = read_samples_from_raw(&mut stdout_reader, &mut buf, 4096)?;
        if n == 0 {
            break;
        }
        let rms_db = compute_rms_dbfs(&buf);
        if rms_db > SILENCE_THRESHOLD_DBFS {
            last_non_silent = Instant::now();
            runtime_silence_warn.store(false, Ordering::SeqCst);
        } else {
            if last_non_silent.elapsed() > Duration::from_secs(RUNTIME_SILENCE_SECONDS) {
                runtime_silence_warn.store(true, Ordering::SeqCst);
            }
        }
        for &s in &buf {
            writer.write_sample(s).ok();
        }

        let active = rms_db > SILENCE_THRESHOLD_DBFS;
        let ind = if active { spinner[spin_i % spinner.len()] } else { "—" };
        print!(
            "\rRECORDING: RUNNING  Book: {}  Chapter: {:02}  Part: {}  Audio: {} ",
            out_path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or(""),
            0,
            0,
            ind
        );
        io::stdout().flush().ok();
        if active {
            spin_i = spin_i.wrapping_add(1);
        }
    }

    writer.finalize().context("finalizing wav")?;
    println!("\nFile saved: {}", out_path.display());
    Ok(())
}

fn pre_record_validation(device: &str) -> Result<()> {
    let (mut child, mut stdout_reader) = spawn_arecord(device)?;
    let samples_needed = (SAMPLE_RATE as usize) * (PRE_RECORD_SECONDS as usize);
    let mut buf: Vec<i32> = Vec::with_capacity(samples_needed);
    let mut total_read = 0usize;
    let start = Instant::now();
    while total_read < samples_needed && start.elapsed() < Duration::from_secs(PRE_RECORD_SECONDS + 1) {
        let n = read_samples_from_raw(&mut stdout_reader, &mut buf, samples_needed - total_read)?;
        total_read += n;
    }
    let _ = child.kill();
    let rms_db = compute_rms_dbfs(&buf);
    if rms_db <= SILENCE_THRESHOLD_DBFS {
        bail!("ERROR: No audio detected (RMS {:.1} dBFS). Check microphone mute button. Recording not started.", rms_db);
    }
    Ok(())
}

fn main() -> Result<()> {
    println!("Rustcorder Phase 1 - minimal TUI (arecord backend)");
    let mut session = load_session();
    session = choose_or_edit_session(session)?;
    session.part = find_next_part(&session.book, session.chapter);
    save_session(&session)?;

    let matches = detect_matching_cards().context("detecting USB audio cards")?;
    if matches.is_empty() {
        bail!("FATAL: No approved USB microphone detected. Expected VID:PID {}:{}.", VID, PID);
    }
    if matches.len() > 1 {
        bail!("FATAL: More than one approved USB microphone detected. Please connect exactly one device.");
    }
    let card = matches[0];
    let device = card_to_alsa_device(card);
    println!("Using ALSA device: {}", device);

    let stop_app = Arc::new(AtomicBool::new(false));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let runtime_silence_warn = Arc::new(AtomicBool::new(false));

    {
        let stop_app = stop_app.clone();
        ctrlc::set_handler(move || {
            stop_app.store(true, Ordering::SeqCst);
        })
        .context("setting ctrlc handler")?;
    }

    loop {
        println!("Ready to record: Book='{}' Chapter={:02} Part={:03}", session.book, session.chapter, session.part);
        println!("Press Enter to start recording part {:03}. Type 'q' then Enter to quit.", session.part);
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.trim().eq_ignore_ascii_case("q") || stop_app.load(Ordering::SeqCst) {
            break;
        }

        let matches = detect_matching_cards()?;
        if matches.len() != 1 {
            println!("FATAL: Approved microphone not present (or multiple). Aborting recording.");
            break;
        }

        match pre_record_validation(&device) {
            Ok(()) => {
                println!("Pre-record validation passed.");
            }
            Err(e) => {
                println!("{}", e);
                continue;
            }
        }

        let out_path = build_output_path(&session.book, session.chapter, session.part)?;
        let (child, stdout_reader) = spawn_arecord(&device)?;
        let stop_flag_clone = stop_flag.clone();
        stop_flag_clone.store(false, Ordering::SeqCst);
        let runtime_warn_clone = runtime_silence_warn.clone();

        let writer_thread = thread::spawn(move || {
            let res = write_wav_stream(stdout_reader, out_path, stop_flag_clone, runtime_warn_clone);
            if let Err(e) = res {
                eprintln!("Recording failed: {:?}", e);
            }
        });

        println!("Recording... press Enter to stop.");
        let mut dummy = String::new();
        io::stdin().read_line(&mut dummy).ok();
        stop_flag.store(true, Ordering::SeqCst);

        writer_thread.join().ok();
        loop {
            print!("Continue chapter? (Y/n): ");
            io::stdout().flush()?;
            let mut resp = String::new();
            io::stdin().read_line(&mut resp)?;
            let resp = resp.trim();
            if resp.is_empty() || resp.eq_ignore_ascii_case("y") {
                session.part += 1;
                save_session(&session)?;
                break;
            } else if resp.eq_ignore_ascii_case("n") {
                session.chapter += 1;
                session.part = 1;
                save_session(&session)?;
                break;
            } else {
                println!("Please answer Y or n.");
            }
        }

        if stop_app.load(Ordering::SeqCst) {
            break;
        }
    }

    println!("Exiting. Saving session.");
    save_session(&session)?;
    Ok(())
}
