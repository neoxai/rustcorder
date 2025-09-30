use clap::{Parser, Subcommand};
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};
use std::path::PathBuf;
use std::fs;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{WavSpec, WavWriter};
use crossterm::{terminal, event::{self, Event, KeyCode, KeyEvent}};

#[derive(Parser)]
#[command(name = "cord")]
#[command(about = "Audio transcription and search tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse audio file to text using Whisper
    Parse {
        /// Input audio file path
        #[arg(long)]
        input: PathBuf,

        /// Path to Whisper model file (e.g., ggml-base.bin)
        #[arg(long, default_value = "models/ggml-base.bin")]
        model: PathBuf,
    },
    /// Record audio from microphone
    Record {
        /// Output filename (optional, defaults to timestamp)
        #[arg(long)]
        output: Option<String>,

        /// Duration in seconds (optional, press Ctrl+C to stop)
        #[arg(long)]
        duration: Option<u64>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Parse { input, model } => {
            parse_audio(&input, &model)?;
        }
        Commands::Record { output, duration } => {
            record_audio(output, duration)?;
        }
    }

    Ok(())
}

fn parse_audio(input_path: &PathBuf, model_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading Whisper model from: {:?}", model_path);

    // Load the Whisper model
    let ctx = WhisperContext::new_with_params(
        model_path.to_str().unwrap(),
        WhisperContextParameters::default()
    )?;

    println!("Transcribing audio file: {:?}", input_path);

    // Create a state
    let mut state = ctx.create_state()?;

    // Set up parameters for transcription
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // Convert audio file to the required format
    // Note: whisper-rs expects 16kHz mono PCM data
    let audio_data = load_audio(input_path)?;

    // Run transcription
    state.full(params, &audio_data)?;

    // Get and print results
    let num_segments = state.full_n_segments()?;
    for i in 0..num_segments {
        let segment = state.full_get_segment_text(i)?;
        println!("{}", segment);
    }

    Ok(())
}

fn load_audio(_path: &PathBuf) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // For now, this is a placeholder
    // You'll need to use a library like `symphonia` or `ffmpeg` to decode the audio
    // and convert it to 16kHz mono f32 PCM format
    Err("Audio loading not yet implemented. Please convert your audio to 16kHz mono WAV first.".into())
}

fn record_audio(output: Option<String>, duration: Option<u64>) -> Result<(), Box<dyn std::error::Error>> {
    // Create recordings directory if it doesn't exist
    fs::create_dir_all("recordings")?;

    // Generate filename
    let filename = match output {
        Some(name) => name,
        None => {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
            format!("recording_{}.wav", timestamp)
        }
    };

    let output_path = PathBuf::from("recordings").join(&filename);
    println!("Recording to: {:?}", output_path);

    // Get the default audio host and input device
    let host = cpal::default_host();
    let device = host.default_input_device()
        .ok_or("No input device available")?;

    println!("Using input device: {}", device.name()?);

    // Get the default input config
    let config = device.default_input_config()?;
    println!("Default input config: {:?}", config);

    // Create WAV writer
    let spec = WavSpec {
        channels: config.channels(),
        sample_rate: config.sample_rate().0,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let writer = Arc::new(Mutex::new(WavWriter::create(&output_path, spec)?));
    let writer_clone = writer.clone();

    // Build the input stream
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut writer = writer_clone.lock().unwrap();
                for &sample in data {
                    let amplitude = (sample * i16::MAX as f32) as i16;
                    writer.write_sample(amplitude).unwrap();
                }
            },
            |err| eprintln!("Error in audio stream: {}", err),
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let mut writer = writer_clone.lock().unwrap();
                for &sample in data {
                    writer.write_sample(sample).unwrap();
                }
            },
            |err| eprintln!("Error in audio stream: {}", err),
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config.into(),
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                let mut writer = writer_clone.lock().unwrap();
                for &sample in data {
                    let normalized = (sample as i32 - 32768) as i16;
                    writer.write_sample(normalized).unwrap();
                }
            },
            |err| eprintln!("Error in audio stream: {}", err),
            None,
        )?,
        _ => return Err("Unsupported sample format".into()),
    };

    // Start recording
    stream.play()?;

    if let Some(secs) = duration {
        println!("Recording for {} seconds...", secs);
        std::thread::sleep(std::time::Duration::from_secs(secs));
    } else {
        println!("Recording... Press Space, Enter, or Ctrl+C to stop.");

        // Enable raw mode to capture key presses
        terminal::enable_raw_mode()?;

        let should_stop = Arc::new(AtomicBool::new(false));
        let should_stop_clone = should_stop.clone();

        // Set up Ctrl+C handler
        ctrlc::set_handler(move || {
            should_stop_clone.store(true, Ordering::SeqCst);
        }).expect("Error setting Ctrl-C handler");

        // Wait for Space, Enter, or Ctrl+C
        loop {
            if should_stop.load(Ordering::SeqCst) {
                break;
            }

            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(KeyEvent { code, .. }) = event::read()? {
                    match code {
                        KeyCode::Char(' ') | KeyCode::Enter | KeyCode::Char('c') => {
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Disable raw mode
        terminal::disable_raw_mode()?;
    }

    // Stop and finalize
    drop(stream);

    // Extract the writer from the Arc<Mutex<>> and finalize
    let writer = Arc::try_unwrap(writer)
        .map_err(|_| "Failed to unwrap Arc")?
        .into_inner()
        .unwrap();
    writer.finalize()?;

    println!("\nRecording saved to: {:?}", output_path);
    Ok(())
}