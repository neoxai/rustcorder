use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::Result;

/// WAV PCM writer fixed at 48 kHz / 24-bit / mono (matches PLAN capture baseline).
///
/// Writes a placeholder header on construction, then patches the byte-count
/// fields on `finalize()`.  Always call `finalize()` before dropping —
/// a dropped-without-finalize file will have zeroed size fields but the
/// audio data itself will still be intact.
pub struct WavWriter {
    writer: BufWriter<File>,
    /// Total number of audio bytes written to the data chunk so far.
    data_bytes: u32,
}

// Fixed capture parameters (matches PLAN baseline).
const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: u16 = 1;
const BITS_PER_SAMPLE: u16 = 24;
const BYTES_PER_SAMPLE: u16 = BITS_PER_SAMPLE / 8; // 3
const BLOCK_ALIGN: u16 = CHANNELS * BYTES_PER_SAMPLE; // 3
const BYTE_RATE: u32 = SAMPLE_RATE * BLOCK_ALIGN as u32; // 144_000

impl WavWriter {
    /// Create a new WAV file at `path` and write the RIFF/fmt header.
    pub fn new(path: &Path) -> Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // ── RIFF chunk ────────────────────────────────────────────────────
        writer.write_all(b"RIFF")?;
        writer.write_all(&0u32.to_le_bytes())?;   // placeholder: file size - 8
        writer.write_all(b"WAVE")?;

        // ── fmt  chunk (16 bytes, PCM) ────────────────────────────────────
        writer.write_all(b"fmt ")?;
        writer.write_all(&16u32.to_le_bytes())?;  // chunk size
        writer.write_all(&1u16.to_le_bytes())?;   // PCM = 1
        writer.write_all(&CHANNELS.to_le_bytes())?;
        writer.write_all(&SAMPLE_RATE.to_le_bytes())?;
        writer.write_all(&BYTE_RATE.to_le_bytes())?;
        writer.write_all(&BLOCK_ALIGN.to_le_bytes())?;
        writer.write_all(&BITS_PER_SAMPLE.to_le_bytes())?;

        // ── data chunk header ─────────────────────────────────────────────
        writer.write_all(b"data")?;
        writer.write_all(&0u32.to_le_bytes())?;   // placeholder: data byte count

        Ok(WavWriter { writer, data_bytes: 0 })
    }

    /// Write S24_LE samples (ALSA stores 24-bit values in the lower 24 bits of
    /// each i32, sign-extended into bits 24-31).
    ///
    /// For WAV 24-bit PCM, each sample is 3 bytes little-endian — that is
    /// exactly bytes [0..3] of each i32's little-endian representation.
    pub fn write_s24le(&mut self, samples: &[i32]) -> Result<()> {
        for &s in samples {
            let bytes = s.to_le_bytes();
            self.writer.write_all(&bytes[..3])?;
            self.data_bytes += 3;
        }
        Ok(())
    }

    /// Flush and patch the RIFF/data size fields.  Must be called to produce a
    /// valid WAV file.
    pub fn finalize(&mut self) -> Result<()> {
        self.writer.flush()?;

        let file = self.writer.get_mut();

        // Byte offset of the data chunk size field:
        //   RIFF(4) + ChunkSize(4) + WAVE(4)       = 12
        //   fmt (4) + ChunkSize(4) + fmtdata(16)   = 24
        //   data(4)                                = 4
        //   Total to reach the data size field: 40
        file.seek(SeekFrom::Start(40))?;
        file.write_all(&self.data_bytes.to_le_bytes())?;

        // RIFF chunk size = 36 (fixed headers after "RIFF????") + data_bytes
        let riff_size: u32 = 36 + self.data_bytes;
        file.seek(SeekFrom::Start(4))?;
        file.write_all(&riff_size.to_le_bytes())?;

        file.flush()?;
        Ok(())
    }
}
