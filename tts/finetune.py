import os
import argparse
import logging
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.config import BaseDatasetConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    logger.info(f"Starting XTTS-v2 fine-tuning for language: {args.target_language}")
    os.makedirs(args.output_path, exist_ok=True)

    # --- 1. Define Dataset Config ---
    # We are fine-tuning on *only* our high-quality single-speaker dataset
    dataset_config = BaseDatasetConfig(
        name=args.target_language,
        meta_file_train=f"{args.target_language}.csv", # e.g., hindi.csv
        path=os.path.join(args.data_path, f"{args.target_language}_single_speaker")
    )

    # --- 2. Load Base XTTS Config ---
    # We load the default config and modify it for fine-tuning
    config = XttsConfig()
    config.load_json(args.base_config_path) # Load the config from the pre-trained model
    
    # Update config for fine-tuning
    config.datasets = [dataset_config]
    config.output_path = args.output_path
    config.run_name = f"xtts_finetune_{args.target_language}"
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.eval_batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    # Set paths to the base model's checkpoint files
    config.model_args.gpt_checkpoint = os.path.join(args.base_model_path, "gpt.pth")
    config.model_args.text_encoder_checkpoint = os.path.join(args.base_model_path, "text_encoder.pth")
    config.model_args.xtts_checkpoint = os.path.join(args.base_model_path, "xtts.pth")

    # --- 3. Initialize Model ---
    logger.info("Initializing XTTS model from pre-trained state...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=args.base_model_path, eval=True)

    # --- 4. Initialize Trainer ---
    logger.info("Initializing Trainer...")
    # TrainingArgs from TTS.trainer
    training_args = TrainingArgs(
        run_name=config.run_name,
        output_path=config.output_path,
        # We are restoring from the base model, so restore_path is set
        restore_path=os.path.join(args.base_model_path, "xtts.pth"), 
        skip_train_epoch=False,
    )

    trainer = Trainer(
        training_args,
        config,
        model=model,
    )

    # --- 5. Start Fine-Tuning ---
    logger.info("Starting fine-tuning. This will take a long time...")
    trainer.fit()
    
    logger.info(f"Fine-tuning complete. Model saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XTTS-v2 Fine-Tuning Script")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the pre-trained XTTS-v2 model directory.")
    parser.add_argument("--base_config_path", type=str, required=True, help="Path to the pre-trained XTTS-v2 config.json.")
    parser.add_argument("--data_path", type=str, required=True, help="Base path to data directories (e.g., 'data/').")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--target_language", type=str, required=True, help="Target language code (e.g., 'hindi').")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for fine-tuning.")
    
    args = parser.parse_args()
    main(args)