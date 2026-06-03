#!/usr/bin/env python3
import argparse

import pandas as pd
import torch
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sdv.utils import load_synthesizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data using CTGAN or CopulaGAN"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        nargs="?",
        default=None,
        help="Path to the input CSV file (required for training)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Only load existing model and sample data (no training)",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to saved model for sampling (if not set, uses --output-model)",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU for training if available"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed training information"
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="synthesizer.pkl",
        help="Path to save the trained model (or load from when sampling)",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="synthetic_data.csv",
        help="Path to save the synthetic data (CSV)",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="Number of rows to generate (default: same as original dataset size for training, or model's metadata for sampling)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (for CTGAN and CopulaGAN)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for training (CTGAN only)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding layer dimension (CTGAN only)",
    )
    parser.add_argument(
        "--generator-dim",
        type=int,
        default=256,
        help="Generator hidden layer dimension (CTGAN only)",
    )
    parser.add_argument(
        "--discriminator-dim",
        type=int,
        default=256,
        help="Discriminator hidden layer dimension (CTGAN only)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Fixed random seed for reproducibility",
    )

    args = parser.parse_args()

    # Sampling mode
    if args.sample:
        # Determine model path
        model_path = args.load_model if args.load_model else args.output_model
        print(f"Loading model from {model_path}...")
        model = load_synthesizer(model_path)

        # Determine number of rows to generate
        if args.num_rows is None:
            # Try to get from model's metadata or default to 100
            try:
                # Some synthesizers store the number of rows used in training
                num_rows = (
                    model._num_rows_trained
                    if hasattr(model, "_num_rows_trained")
                    else 100
                )
            except:
                num_rows = 100
            print(
                f"No --num-rows provided, using {num_rows} as default (set by model or fallback)"
            )
        else:
            num_rows = args.num_rows

        print(f"Generating {num_rows} synthetic rows...")
        synthetic_data = model.sample(num_rows=num_rows)

        print(f"Saving synthetic data to {args.output_data}...")
        synthetic_data.to_csv(args.output_data, index=False)
        print("Done!")
        return

    # Training mode (requires input_csv)
    if args.input_csv is None:
        parser.error(
            "input_csv is required for training mode (unless --sample is used)"
        )

    # Set seed for reproducibility
    if args.random_seed is not None:
        import random

        import numpy as np

        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available() and args.gpu:
            torch.cuda.manual_seed_all(args.random_seed)

    # 1. Load data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"Data shape: {df.shape}")

    # 2. Detect metadata automatically
    print("Detecting metadata...")
    metadata = Metadata.detect_from_dataframe(data=df, table_name="my_table")

    # 3. Create and configure the model
    print(f"Creating {args.model} model...")
    model_kwargs = {
        "metadata": metadata,
        "verbose": args.verbose,
        "enforce_min_max_values": True,
        "enforce_rounding": False,
    }

    model_kwargs.update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "embedding_dim": args.embedding_dim,
            "generator_dim": (args.generator_dim, args.generator_dim),
            "discriminator_dim": (args.discriminator_dim, args.discriminator_dim),
            "cuda": args.gpu and torch.cuda.is_available(),
        }
    )
    model = CTGANSynthesizer(**model_kwargs)

    # 4. Train the model
    print("Training model...")
    model.fit(df)

    # 5. Save the model
    print(f"Saving model to {args.output_model}...")
    model.save(filepath=args.output_model)

    # 6. Generate synthetic data
    num_rows = args.num_rows if args.num_rows is not None else len(df)
    print(f"Generating {num_rows} synthetic rows...")
    synthetic_data = model.sample(num_rows=num_rows)

    # 7. Save synthetic data
    print(f"Saving synthetic data to {args.output_data}...")
    synthetic_data.to_csv(args.output_data, index=False)
    print("Done!")


if __name__ == "__main__":
    main()

###########################################################################################
# python3 ctgan_generator.py data.csv \
#   --output-model /home/kkatsuba/work_flow/dss_generation/trained_models/ctgan_data.pkl \
#   --gpu \
#   --verbose \
#   --output-data /home/kkatsuba/work_flow/dss_generation/synth_data/synth_data.csv


# python3 ctgan_generator.py data.csv \
#   --sample
#   --load-model /home/kkatsuba/work_flow/dss_generation/trained_models/ctgan_data.pkl \
#   --output-data /home/kkatsuba/work_flow/dss_generation/synth_data/synth_data.csv
###########################################################################################
