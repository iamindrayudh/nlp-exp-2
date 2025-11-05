import os
import random
import pandas as pd
from TTS.api import TTS
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_tts_data(text_file_path, speaker_wav_dir, output_dir, language_code):
    logger.info("Loading base XTTS-v2 model...")
    
    # Load the base XTTS-v2 model from Coqui
    # This will download it if not present
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    logger.info("XTTS-v2 model loaded successfully.")

    
    logger.info(f"Loading texts from {text_file_path}...")
    try:
        sentences = pd.read_csv(text_file_path, sep="\t")["sentence"].dropna().tolist()
    except Exception as e:
        logger.error(f"Could not read text file: {e}. Exiting.")
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
        
    logger.info(f"Starting GEN_TTS generation for {len(sentences)} sentences...")
    
    output_metadata = []
    
    for i, text in enumerate(tqdm(sentences, desc="Generating GEN_TTS")):
        # Pick a random English speaker for zero-shot synthesis
        speaker_wav = random.choice(speaker_wavs)
        output_wav_path = os.path.join(output_dir, f"gen_tts_{i}.wav")
        
        try:
            # Synthesize the Hindi text using the English speaker's voice
            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language_code, # e.g., 'hi' or 'bn'
                file_path=output_wav_path
            )
            output_metadata.append({"audio_path": output_wav_path, "text": text})
            
        except Exception as e:
            logger.warning(f"Failed to synthesize text: {text} | Speaker: {speaker_wav} | Error: {e}")

    # --- Save metadata for the ASR training ---
    metadata_path = os.path.join(output_dir, "gen_tts_metadata.csv")
    metadata_df = pd.DataFrame(output_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    
    logger.info(f"GEN_TTS generation complete. Metadata saved to {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEN_TTS Data Generation Script (XTTS-v2)")
    parser.add_argument("--text_file_path", type=str, required=True, help="Path to the .csv or .tsv file containing sentences.")
    parser.add_argument("--speaker_wav_dir", type=str, required=True, help="Directory containing English speaker reference .wav files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated audio and metadata.")
    parser.add_argument("--language", type=str, required=True, help="Language code for synthesis (e.g., 'hi', 'bn').")
    
    args = parser.parse_args()
    generate_tts_data(args.text_file_path, args.speaker_wav_dir, args.output_dir, args.language)