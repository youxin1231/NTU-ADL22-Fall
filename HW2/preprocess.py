from pathlib import Path
import argparse
import json

TRAIN = "train"
DEV = "valid"
SPLITS = [TRAIN, DEV]


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--data_dir", type=Path, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None, help="A directory where output files are stored."
    )

    args = parser.parse_args()
    return args


def preprocess(data, context):
    preprocessed = []
    for i in range(len(data)):
        d = {}
        d['video-id'] = data[i]['id']
        d['fold-ind'] = i
        d['startphrase'] = data[i]['question']
        d['sent1'] = data[i]['question']
        d['sent2'] = ''
        d['gold-source'] = 'gold'
        d['ending0'] = context[data[i]['paragraphs'][0]]
        d['ending1'] = context[data[i]['paragraphs'][1]]
        d['ending2'] = context[data[i]['paragraphs'][2]]
        d['ending3'] = context[data[i]['paragraphs'][3]]
        for j in range(4):
            if(data[i]['relevant'] == data[i]['paragraphs'][j]):
                d['relevant'] = j 
        preprocessed.append(d)
    return preprocessed


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

    context_path = args.data_dir / f"context.json"
    context = json.loads(context_path.read_text())

    for split in SPLITS:
        preprocessed_file = preprocess(data[split], context)
        
        output_path = args.output_dir / f"{split}.json"
        with open(str(output_path), 'w', encoding='UTF-8') as f:
            json.dump(preprocessed_file, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()