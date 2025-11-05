import os
import random
import pandas as pd
from TTS.api import TTS
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_vc_data(target_audio_metadata, speaker_wav_dir, output_dir, num_conversions=5):
    logger.info("Loading base XTTS-v2 model for Voice Conversion...")
    
    # Load the base XTTS-v2 model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    logger.info("XTTS-v2 model loaded successfully.")

    logger.info(f"Loading target audio metadata from {target_audio_metadata}...")
    try:
        target_data = pd.read_csv(target_audio_metadata)
        if "audio_path" not in target_data.columns or "text" not in target_data.columns:
            logger.error("Metadata must contain 'audio_path' and 'text' columns.")
            return
    except Exception as e:
        logger.error(f"Could not read metadata: {e}. Exiting.")
        return

    logger.info(f"Loading English speaker reference wavs from {speaker_wav_dir}...")
    try:
        speaker_wavs = [os.path.join(speaker_wav_dir, f) for f in os.listdir(speaker_wav_dir) if f.endswith((".wav", ".mp3", ".flac"))]
        if not speaker_wavs:
            raise FileNotFoundError
    except FileNotFoundError:
        logger.error(f"No speaker wavs found in {speaker_wav_dir}. Cannot proceed.")
        return

    os.makedirs(output_dir, exist_ok=True)
        
    logger.info(f"Starting GEN_VC generation for {len(target_data)} audio files...")
    
    output_metadata = []
    
    for _, row in tqdm(target_data.iterrows(), total=len(target_data), desc="Generating GEN_VC"):
        source_wav_path = row["audio_path"]
        text = row["text"]
        
        if not os.path.exists(source_wav_path):
            logger.warning(f"Source file not found, skipping: {source_wav_path}")
            continue
        
        # Select N random English speakers to convert to
        target_speakers = random.sample(speaker_wavs, min(num_conversions, len(speaker_wavs)))
        
        for j, speaker_wav in enumerate(target_speakers):
            base_name = os.path.splitext(os.path.basename(source_wav_path))[0]
            output_wav_path = os.path.join(output_dir, f"gen_vc_{base_name}_{j}.wav")
            
            try:
                # Perform cross-lingual voice conversion
                tts.voice_conversion_to_file(
                    source_wav=source_wav_path,
                    target_wav=speaker_wav,
                    file_path=output_wav_path
                )
                output_metadata.append({"audio_path": output_wav_path, "text": text})
                
            except Exception as e:
                logger.warning(f"Failed to convert: {source_wav_path} to {speaker_wav} | Error: {e}")

    # --- Save metadata for the ASR training ---
    metadata_path = os.path.join(output_dir, "gen_vc_metadata.csv")
    metadata_df = pd.DataFrame(output_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    
    logger.info(f"GEN_VC generation complete. Metadata saved to {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEN_VC Data Generation Script (XTTS-v2)")
    parser.add_argument("--target_audio_metadata", type=str, required=True, help="Path to the .csv file of the single-speaker target audio.")
    parser.add_argument("--speaker_wav_dir", type=str, required=True, help="Directory containing English speaker reference .wav files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated audio and metadata.")
    parser.add_argument("--num_conversions", type=int, default=5, help="Number of different speakers to convert each file to.")
    
    args = parser.parse_args()
    generate_vc_data(args.target_audio_metadata, args.speaker_wav_dir, args.output_dir, args.num_conversions)