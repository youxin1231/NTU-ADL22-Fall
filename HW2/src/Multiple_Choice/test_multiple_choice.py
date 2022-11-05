import argparse
from pathlib import Path
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the test data on a multiple choice task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ckpt/bert-base-chinese/multiple_choice",
        help="Path to pretrained model or model identifier from huggingface.co/models."
        )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default="data/preprocessed/test_swag.json",
        help="A csv or a json file containing the training data."
        )
    parser.add_argument(
        "--output_file", 
        type=Path, 
        default="data/pred/multiple_choice_pred.json",
        help="A csv or a json file containing the training data."
        )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=args.test_file)
    

if __name__ == "__main__":
    main()