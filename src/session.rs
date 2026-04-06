use std::fs;
use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;

const SESSION_FILE: &str = "session.save.txt";

/// Bytes per second for our fixed capture format (48 kHz, mono, 24-bit).
const BYTE_RATE: f64 = 144_000.0;

/// Persistent recording session: book name, current chapter, current part,
/// and the absolute timeline position (in seconds) within the current chapter.
#[derive(Debug, Clone)]
pub struct Session {
    pub book: String,
    pub chapter: u32,
    pub part: u32,
    /// Absolute timeline position (seconds from chapter start) where the next
    /// recording clip should begin.  Updated after each clip stops or a
    /// punch-in point is confirmed.
    pub timeline_pos: f64,
}

impl Session {
    pub fn new(book: String, chapter: u32) -> Self {
        Session { book, chapter, part: 1, timeline_pos: 0.0 }
    }

    /// Directory where recordings for this book are stored: `$PWD/<book>/`
    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(&self.book)
    }

    /// Full path for the current part file: `<book>/Chapter_NN_partNNN.wav`
    pub fn current_path(&self) -> PathBuf {
        self.output_dir().join(self.current_filename())
    }

    pub fn current_filename(&self) -> String {
        format!("Chapter_{:02}_part{:03}.wav", self.chapter, self.part)
    }

    /// Move to the next part within the same chapter.
    pub fn advance_part(&mut self) {
        self.part += 1;
    }

    /// Mark the chapter complete: bump chapter, reset part and timeline position.
    pub fn advance_chapter(&mut self) {
        self.chapter += 1;
        self.part = 1;
        self.timeline_pos = 0.0;
    }

    /// Ensure the output directory exists.
    pub fn ensure_output_dir(&self) -> Result<()> {
        fs::create_dir_all(self.output_dir())?;
        Ok(())
    }

    /// Return a path that is guaranteed not to already exist, incrementing
    /// the part counter as needed. Updates `self.part` in place.
    pub fn next_free_path(&mut self) -> PathBuf {
        while self.current_path().exists() {
            self.part += 1;
        }
        self.current_path()
    }

    /// Compute the duration in seconds from a raw WAV data byte count.
    pub fn duration_from_bytes(data_bytes: u32) -> f64 {
        data_bytes as f64 / BYTE_RATE
    }

    /// Remove the last CLIP entry from this chapter's timeline index file.
    ///
    /// Used when a clip is discarded after recording (e.g. DISCARD_SHORT_CLIPS).
    /// The entry is already in the file because it was written at recording start
    /// for crash-safety; this retracts it at stop time.
    pub fn remove_last_timeline_entry(&self) -> Result<()> {
        let path = self
            .output_dir()
            .join(format!("Chapter_{:02}_timeline.txt", self.chapter));

        if !path.exists() {
            return Ok(());
        }

        let content = fs::read_to_string(&path)?;
        let lines: Vec<&str> = content.lines().collect();

        // Find the last line that starts with "CLIP ".
        if let Some(idx) = lines.iter().rposition(|l| l.starts_with("CLIP ")) {
            let new_content = lines
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, l)| *l)
                .collect::<Vec<_>>()
                .join("\n");
            // Preserve trailing newline.
            let new_content = if content.ends_with('\n') {
                format!("{}\n", new_content)
            } else {
                new_content
            };
            fs::write(&path, new_content)?;
        }

        Ok(())
    }

    /// Append a CLIP entry to this chapter's timeline index file.
    ///
    /// The file is created with a header comment on first write.  Entries are
    /// always appended — never overwritten — so a crash mid-recording still
    /// leaves a valid (if short) entry.
    pub fn append_timeline_entry(&self, filename: &str, start: f64) -> Result<()> {
        let path = self
            .output_dir()
            .join(format!("Chapter_{:02}_timeline.txt", self.chapter));
        let is_new = !path.exists();
        let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)?;
        if is_new {
            writeln!(
                file,
                "# rustcorder timeline — {} / Chapter {:02}",
                self.book, self.chapter
            )?;
        }
        writeln!(file, "CLIP {} START={:.3}", filename, start)?;
        Ok(())
    }
}

// ── Persistence ──────────────────────────────────────────────────────────────

pub fn load() -> Option<Session> {
    let content = fs::read_to_string(SESSION_FILE).ok()?;

    let mut book = None;
    let mut chapter: Option<u32> = None;
    let mut part: Option<u32> = None;
    let mut timeline_pos: f64 = 0.0;

    for line in content.lines() {
        if let Some(v) = line.strip_prefix("BOOK=") {
            book = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("CHAPTER=") {
            chapter = v.trim().parse().ok();
        } else if let Some(v) = line.strip_prefix("PART=") {
            part = v.trim().parse().ok();
        } else if let Some(v) = line.strip_prefix("TIMELINE_POS=") {
            timeline_pos = v.trim().parse().unwrap_or(0.0);
        }
    }

    Some(Session {
        book: book?,
        chapter: chapter?,
        part: part?,
        timeline_pos,
    })
}

pub fn save(session: &Session) -> Result<()> {
    let content = format!(
        "BOOK={}\nCHAPTER={:02}\nPART={}\nTIMELINE_POS={:.3}\n",
        session.book, session.chapter, session.part, session.timeline_pos
    );
    fs::write(SESSION_FILE, content)?;
    Ok(())
}
