🧠 Core Concept (Crossfade Math)

Let:

A[i] = sample from first audio
B[i] = sample from second audio
t = normalized position in fade (0.0 → 1.0)

Then:

output[i] = A[i] * (1.0 - t) + B[i] * t

That’s a linear crossfade.

⚙️ Step-by-Step in Rust
1. Use a WAV crate

The most common one:

hound

Add to Cargo.toml:

hound = "3"
2. Read WAV files
use hound;

fn read_wav(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap();
    reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect()
}
3. Crossfade function
fn crossfade(a: &[f32], b: &[f32], fade_len: usize) -> Vec<f32> {
    let mut output = Vec::new();

    let a_len = a.len();
    let b_len = b.len();

    let fade_start = a_len - fade_len;

    // 1. Copy first part of A
    output.extend_from_slice(&a[..fade_start]);

    // 2. Crossfade region
    for i in 0..fade_len {
        let t = i as f32 / fade_len as f32;

        let a_sample = a[fade_start + i];
        let b_sample = b[i];

        let mixed = a_sample * (1.0 - t) + b_sample * t;
        output.push(mixed);
    }

    // 3. Copy remainder of B
    output.extend_from_slice(&b[fade_len..]);

    output
}
4. Write output WAV
fn write_wav(path: &str, samples: &[f32], spec: hound::WavSpec) {
    let mut writer = hound::WavWriter::create(path, spec).unwrap();

    for &sample in samples {
        let s = (sample * i16::MAX as f32) as i16;
        writer.write_sample(s).unwrap();
    }

    writer.finalize().unwrap();
}
🎚️ Better Sound: Use Equal Power Crossfade

Linear fades can sound like a dip in volume.

Instead use:

A_gain = cos(t * π/2)
B_gain = sin(t * π/2)

Replace this line:

let mixed = a_sample * (1.0 - t) + b_sample * t;

With:

let a_gain = (1.0 - t).cos(); // approximation
let b_gain = t.sin();

let mixed = a_sample * a_gain + b_sample * b_gain;

(More correctly:)

let a_gain = (std::f32::consts::FRAC_PI_2 * (1.0 - t)).cos();
let b_gain = (std::f32::consts::FRAC_PI_2 * t).sin();
⚠️ Important Gotchas
1. Sample Rate MUST Match

If not, resample first (e.g. 44.1kHz vs 48kHz).

2. Channel Count

If stereo:

Interleave handling required (L,R,L,R,...)
Or split into channels and process separately
3. Clipping

After summing:

let mixed = mixed.clamp(-1.0, 1.0);
🧩 Useful Rust Audio Crates
hound – WAV I/O