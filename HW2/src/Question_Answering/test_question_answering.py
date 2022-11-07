import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
)
from utils_qa import postprocess_qa_predictions
from tqdm import tqdm
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Predict the test data on a question answering task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ckpt/bert-base-chinese/multiple_choice",
        help="Path to pretrained model or model identifier from huggingface.co/models."
        )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default="data/multiple_choice_pred.json",
        help="A csv or a json file containing the training data."
        )
    parser.add_argument(
        "--pred_file", 
        type=Path, 
        default="data/pred/result.csv",
        help="A csv or a json file containing the training data."
        )

    # Data
    parser.add_argument("--max_length", type=int, default=512)

    # Data loader
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
    )
    model.to(args.device)

    def prepare_test_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    # Dataset creation:
    data_files = {}
    data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    column_names = raw_datasets["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(args.max_length, tokenizer.model_max_length)

    predict_examples = raw_datasets["test"]
    predict_dataset = predict_examples.map(
                prepare_test_features,
                batched=True,
                remove_columns=column_names
    )

    # DataLoaders creation:
    data_collator = default_data_collator
    predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
    predict_dataloader = DataLoader(
        predict_dataset_for_model, collate_fn=data_collator, batch_size=args.batch_size
    )

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        return formatted_predictions


    # Predict
    print("***** Running Question Answering Prediction *****")
    print(f"Num examples = {len(predict_dataset)}")
    print(f"Batch size = {args.batch_size}")

    all_start_logits = []
    all_end_logits = []
    
    model.eval()
    
    for step, batch in enumerate(tqdm(predict_dataloader)):
        batch["input_ids"] = batch["input_ids"].to(args.device)
        batch["attention_mask"] = batch["attention_mask"].to(args.device)
        batch["token_type_ids"] = batch["token_type_ids"].to(args.device)
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            all_start_logits.append(start_logits.cpu())
            all_end_logits.append(end_logits.cpu())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)

    args.pred_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.pred_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['id', 'answer'])
        for i in range(len(prediction)):
            writer.writerow([prediction[i]['id'], prediction[i]['prediction_text']])

if __name__ == "__main__":
    main()