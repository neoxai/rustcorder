use std::fs;
use std::path::PathBuf;

use anyhow::Result;

const SESSION_FILE: &str = "session.save.txt";

/// Persistent recording session: book name, current chapter, current part.
#[derive(Debug, Clone)]
pub struct Session {
    pub book: String,
    pub chapter: u32,
    pub part: u32,
}

impl Session {
    pub fn new(book: String, chapter: u32) -> Self {
        Session { book, chapter, part: 1 }
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

    /// Mark the chapter complete: bump chapter, reset part to 1.
    pub fn advance_chapter(&mut self) {
        self.chapter += 1;
        self.part = 1;
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
}

// ── Persistence ──────────────────────────────────────────────────────────────

pub fn load() -> Option<Session> {
    let content = fs::read_to_string(SESSION_FILE).ok()?;

    let mut book = None;
    let mut chapter: Option<u32> = None;
    let mut part: Option<u32> = None;

    for line in content.lines() {
        if let Some(v) = line.strip_prefix("BOOK=") {
            book = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("CHAPTER=") {
            chapter = v.trim().parse().ok();
        } else if let Some(v) = line.strip_prefix("PART=") {
            part = v.trim().parse().ok();
        }
    }

    Some(Session {
        book: book?,
        chapter: chapter?,
        part: part?,
    })
}

pub fn save(session: &Session) -> Result<()> {
    let content = format!(
        "BOOK={}\nCHAPTER={:02}\nPART={}\n",
        session.book, session.chapter, session.part
    );
    fs::write(SESSION_FILE, content)?;
    Ok(())
}
