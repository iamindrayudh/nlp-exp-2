import os
import sys
import argparse
import json
import logging
import pandas as pd
import numpy as np
import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor
)
from datasets import load_dataset, load_metric, Dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import librosa

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Data Collator ---
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding with -100 to be ignored by loss function
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

# --- 2. Metrics Computation ---
def compute_metrics(pred):
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

# --- 3. Audio & Text Prep Functions ---
def speech_file_to_array(path):
    y, sr = librosa.load(path, sr=16000)
    return y

def prepare_dataset(batch):
    batch["input_values"] = speech_file_to_array(batch["audio_path"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

# --- 4. Main Training Function ---
def main(args):
    logger.info(f"Starting ASR training for experiment: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Processor and Model ---
    logger.info(f"Loading base model: {args.model_name_or_path}")
    
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model_name_or_path)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    
    global processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name_or_path,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_extractor()

    # --- Load and Prepare Dataset ---
    logger.info(f"Loading and preprocessing dataset from {args.dataset_path}")
    
    df = pd.read_csv(args.dataset_path)
    df = df.sample(frac=1).reset_index(drop=True)
    split_index = int(len(df) * 0.9)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    train_df.to_csv(os.path.join(args.output_dir, "train_split.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test_split.csv"), index=False)

    train_dataset = Dataset.from_pandas(train_df).map(prepare_dataset, remove_columns=train_df.columns)
    eval_dataset = Dataset.from_pandas(test_df).map(prepare_dataset, remove_columns=test_df.columns)

    logger.info(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples.")

    # --- Data Collator ---
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=100,
    )
    
    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )
    
    # --- Train ---
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    # --- Evaluate ---
    logger.info("Running final evaluation...")
    results = trainer.evaluate()
    
    # --- Save Results ---
    results_path = os.path.join(args.output_dir, "final_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Final evaluation results: {results}")
    logger.info(f"Training pipeline complete. Results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Training Script")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pre-trained model (e.g., 'ai4bharat/indic-wav2vec-hindi').")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the master metadata CSV file (e.g., 'data/metadata/hindi.csv').")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and results.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate.")
    
    args = parser.parse_args()
    main(args)