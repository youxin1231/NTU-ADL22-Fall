from pathlib import Path
import argparse
import json

TRAIN = "train"
DEV = "valid"
SPLITS = [TRAIN, DEV]


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing the train / validation / test data.")
    parser.add_argument(
        "--data_dir", type=Path, default=None, help="A directory containing the training data."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None, help="A directory where output files are stored."
    )


    parser.add_argument(
        "--test_preprocess", action="store_true", default=False, help="Whether is test preprocessing or not."
    )
    parser.add_argument(
        "--test_file", type=Path, default=None, help="Path to the test file."
    )
    parser.add_argument(
        "--context_file", type=Path, default=None, help="Path to the conext file."
    )
    parser.add_argument(
        "--output_file", type=Path, default=None, help="Path to the output file."
    )
    args = parser.parse_args()
    return args


def preprocess_swag(data, context):
    swag = []

    for i in range(len(data)):
        # SWAG dataset format
        d = {}

        d['video-id'] = data[i]['id']
        d['fold-ind'] = str(0)
        d['startphrase'] = data[i]['question']

        d['sent1'] = data[i]['question']
        d['sent2'] = ''

        d['gold-source'] = 'gold'

        d['ending0'] = context[data[i]['paragraphs'][0]]
        d['ending1'] = context[data[i]['paragraphs'][1]]
        d['ending2'] = context[data[i]['paragraphs'][2]]
        d['ending3'] = context[data[i]['paragraphs'][3]]

        if not args.test_preprocess:
            for j in range(4):
                if(data[i]['relevant'] == data[i]['paragraphs'][j]):
                    d['label'] = j

        swag.append(d)
    return swag

def preprocess_squad(data, context):
    squad = []

    for i in range(len(data)):
        # SQuAD dataset format
        d = {}

        d['answers'] = {}
        d['answers']['answer_start'] = [data[i]['answer']['start']]
        d['answers']['text'] = [data[i]['answer']['text']]

        d['context'] = context[data[i]['relevant']]
        d['id'] = data[i]['id']

        d['question'] = data[i]['question']
        d['title'] = 'train'

        squad.append(d)
    return squad


def main():
    if args.test_preprocess:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)

        data = json.loads(args.test_file.read_text())
        context = json.loads(args.context_file.read_text())

        swag = preprocess_swag(data, context)
        
        args.output_file.write_text(json.dumps(swag, indent=2, ensure_ascii=False, allow_nan=False), encoding='UTF-8')

    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
        data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

        context_path = args.data_dir / f"context.json"
        context = json.loads(context_path.read_text())

        for split in SPLITS:
            swag  = preprocess_swag(data[split], context)
            squad = preprocess_squad(data[split], context)
            
            swag_path = args.output_dir / f"{split}_swag.json"
            swag_path.write_text(json.dumps(swag, indent=2, ensure_ascii=False, allow_nan=False), encoding='UTF-8')

            squad_path = args.output_dir / f"{split}_squad.json"
            squad_path.write_text(json.dumps(squad, indent=2, ensure_ascii=False, allow_nan=False), encoding='UTF-8')
                

if __name__ == "__main__":
    args = parse_args()
    main()