"""
Hugging Face Dataset Preparation Script.
Converts the processed 52,124 customer support intent dataset into Parquet format
with metadata for reproducibility and external hosting on a private Hugging Face Dataset repository.
"""

import json
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_CSV = BASE_DIR / "amazon_final_intent_dataset.csv"
OUTPUT_DIR = BASE_DIR / "hf_dataset"
DATA_DIR = OUTPUT_DIR / "data"
META_DIR = OUTPUT_DIR / "metadata"


def prepare_hf_dataset():
    if not INPUT_CSV.exists():
        logger.error(f"Input dataset not found at {INPUT_CSV}")
        return

    logger.info(f"Loading full dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df):,} total samples.")

    # Create export directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # Deterministic train/test split (85/15) matching historical baseline
    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df["intent"] if "intent" in df.columns else None
    )

    logger.info(f"Split sizes: Train={len(train_df):,}, Test={len(test_df):,}")

    # Export Parquet files
    full_parquet = DATA_DIR / "amazon_final_intent_dataset.parquet"
    train_parquet = DATA_DIR / "train.parquet"
    test_parquet = DATA_DIR / "test.parquet"

    df.to_parquet(full_parquet, index=False, engine="pyarrow")
    train_df.to_parquet(train_parquet, index=False, engine="pyarrow")
    test_df.to_parquet(test_parquet, index=False, engine="pyarrow")
    logger.info(f"Exported Parquet data to {DATA_DIR}")

    # Export label mapping
    intents = sorted(df["intent"].unique().tolist())
    label_map = {intent: idx for idx, intent in enumerate(intents)}
    label_map_file = META_DIR / "label_map.json"
    with open(label_map_file, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    # Export dataset summary metadata
    summary = {
        "dataset_name": "amazon-customer-support-intents",
        "description": "52,124 multi-turn customer support conversations from AmazonHelp Twitter customer service",
        "total_samples": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "num_classes": len(intents),
        "classes": intents,
        "class_distribution": df["intent"].value_counts().to_dict(),
        "source": "Twitter Customer Support Dataset (TWCS)",
        "recommended_visibility": "private",
        "license_rationale": "TWCS contains customer-company interactions governed by Twitter Developer Terms and Kaggle CC BY-NC-SA 4.0. Private HF repository is recommended for corporate compliance."
    }
    summary_file = META_DIR / "dataset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Dataset preparation complete! Artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_hf_dataset()
