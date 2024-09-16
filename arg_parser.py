import argparse

def parse_args():
    """
    Takes parameter from user for training
    :return: Inputs
    """
    # print("Inside: parse_args()")
    parser = argparse.ArgumentParser(
        description="Fine-tune Transformer-based model on Turkish Stance Detection Dataset."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        help="Learning Rate parameter for the Model.",
        type=float
    )
    parser.add_argument(
        "--weight_decay",
        default=0.003,
        help="Weight Decay parameter for the Model.",
        type=float
    )
    parser.add_argument(
        "--model_name",
        default="dbmdz/bert-base-turkish-cased",
        help="Name of the transformer-based model.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        help="Batch size for training.",
        type=int
    )
    parser.add_argument(
        "--epoch",
        default=10,
        help="Step for training.",
        type=int
    )
    parser.add_argument(
        "--max_len",
        default=512,
        help="Maximum token length for model.",
        type=int
    )
    parser.add_argument(
        "--save_dir",
        default="trained-models",
        help="Path for saving trained models.",
        type=str
    )
    parser.add_argument(
        "--checkpoint",
        default="stance-chkpt",
        help="Logging directory for checkpoints.",
        type=str
    )
    parser.add_argument(
        "--hf_repo_name",
        default="",
        help="Hugginface repository name to push model or tokenizer",
        type=str
    )

    return parser.parse_args()