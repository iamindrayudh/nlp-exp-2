import os
import argparse
import json
import logging
import pandas as pd
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_metric
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    logger.info(f"Starting evaluation for model: {args.model_path}")

    # --- 1. Load Model and Processor ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        processor = Wav2Vec2Processor.from_pretrained(args.model_path)
        model = Wav2Vec2ForCTC.from_pretrained(args.model_path)
        model.to(device)
        model.eval()
        logger.info("Model and processor loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load model from {args.model_path}. Exiting. Error: {e}")
        return

    # --- 2. Load Test Data ---
    logger.info(f"Loading test data from: {args.dataset_path}")
    try:
        test_df = pd.read_csv(args.dataset_path)
        if "audio_path" not in test_df.columns or "text" not in test_df.columns:
            logger.error("Metadata must contain 'audio_path' and 'text' columns.")
            return
    except Exception as e:
        logger.critical(f"Could not read metadata: {e}. Exiting.")
        return

    # --- 3. Load Metrics ---
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    predictions = []
    references = []

    # --- 4. Run Inference ---
    logger.info(f"Running inference on {len(test_df)} test samples...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        try:
            # Load and resample audio
            audio_input, sample_rate = librosa.load(row["audio_path"], sr=16000)
            
            # Process audio
            inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
            
            # Run inference
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
            
            # Get prediction
            pred_ids = torch.argmax(logits, dim=-1)
            pred_text = processor.batch_decode(pred_ids)[0]
            
            # Get reference
            ref_text = row["text"]

            predictions.append(pred_text)
            references.append(ref_text)
            
        except Exception as e:
            logger.warning(f"Failed to process file {row['audio_path']}: {e}. Skipping.")

    # --- 5. Compute Final Metrics ---
    logger.info("Computing final metrics...")
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    results = {
        "wer": wer,
        "cer": cer,
        "num_test_samples": len(test_df),
        "num_processed_samples": len(predictions)
    }

    # --- 6. Save Results ---
    results_path = os.path.join(args.model_path, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Evaluation complete. Results: {results}")
    logger.info(f"Evaluation report saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone ASR Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned ASR model directory (e.g., 'outputs/final_asr_hindi').")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the test metadata CSV file.")
    
    args = parser.parse_args()
    main(args)