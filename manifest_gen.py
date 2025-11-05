import os
import argparse
import pandas as pd
from tqdm import tqdm
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_transcripts(root_dir, transcript_extension=".txt"):
    """Finds all transcript files and parses them into a dictionary."""
    transcripts = {}
    logger.info(f"Searching for transcript files in {root_dir}...")
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(transcript_extension):
                file_path = os.path.join(subdir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            # Assumes a format like "file_id_001 This is the text"
                            parts = line.strip().split(' ', 1)
                            if len(parts) == 2:
                                file_id, text = parts
                                # Normalize text (remove punctuation, lower)
                                text = re.sub(r'[^\w\s]', '', text).lower()
                                transcripts[file_id] = text
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")
    logger.info(f"Found {len(transcripts)} transcript entries.")
    return transcripts

def create_manifest(data_dir, output_file, transcripts):
    """Walks a directory to find audio files and match them with transcripts."""
    logger.info(f"Matching audio files in {data_dir} with transcripts...")
    manifest = []
    
    for subdir, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".wav", ".mp3", ".flac")):
                file_id = os.path.splitext(file)[0]
                if file_id in transcripts:
                    audio_path = os.path.join(subdir, file)
                    text = transcripts[file_id]
                    manifest.append({
                        "audio_path": os.path.abspath(audio_path),
                        "text": text
                    })
                else:
                    logger.warning(f"No transcript found for audio file: {file}")

    if not manifest:
        logger.error(f"No audio/transcript pairs were found in {data_dir}. Manifest will be empty.")
        return

    # Save the manifest
    df = pd.DataFrame(manifest)
    df.to_csv(output_file, index=False)
    logger.info(f"Successfully generated manifest with {len(df)} entries. Saved to {output_file}")

def main(args):
    # This structure implies a "raw" data layout, which is very realistic
    # e.g., data/raw/hindi/audio/ and data/raw/hindi/transcripts/
    raw_audio_dir = os.path.join(args.data_base_path, args.language, "audio")
    raw_transcript_dir = os.path.join(args.data_base_path, args.language, "transcripts")
    
    output_manifest_path = os.path.join(args.output_dir, f"{args.language}.csv")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Find all transcripts
    transcripts = find_transcripts(raw_transcript_dir)
    
    # 2. Find all audio files and map them
    if transcripts:
        create_manifest(raw_audio_dir, output_manifest_path, transcripts)
    else:
        logger.error("No transcripts were loaded. Cannot create manifest.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Manifest Generation Script")
    parser.add_argument("--data_base_path", type=str, default="data/raw/", help="Base path to the raw, unprocessed dataset directories.")
    parser.add_argument("--output_dir", type=str, default="data/metadata/", help="Directory to save the final .csv manifest files.")
    parser.add_argument("--language", type=str, required=True, choices=['hindi', 'bengali', 'english'], help="Language to process.")
    
    args = parser.parse_args()
    main(args)