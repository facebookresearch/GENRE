import os
import argparse
import logging
import jsonlines

from tqdm import tqdm
from genre.utils import create_input


def convert_kilt_to_fairseq(input_folder, output_folder, dataset_name, split_name):

    dataset_fairseq = []
    with jsonlines.open(
        os.path.join(input_folder, "{}-{}-kilt.jsonl".format(dataset_name, split_name))
    ) as f:
        for doc in tqdm(
            f, desc="Processing"
        ):
            for title in set(
                prov["title"]
                for out in doc["output"]
                if "provenance" in out
                for prov in out["provenance"]
                if prov.get("bleu_score", 1) > 0.5
            ):
                dataset_fairseq.append((create_input(doc, max_length=384), title))
                if "meta" in doc and "template_questions" in doc["meta"]:
                    for template_question in doc["meta"]["template_questions"]:
                        dataset_fairseq.append((template_question, title))

    if not os.path.exists(
        os.path.join(
            output_folder,
            dataset_name,
        )
    ):
        os.mkdir(
            os.path.join(
                output_folder,
                dataset_name,
            )
        )

    with open(
        os.path.join(
            output_folder,
            dataset_name,
            "{}.source".format(split_name),
        ),
        "w",
    ) as f:
        f.writelines(
            [
                doc[0].replace("\r", ">>").replace("\n", ">>") + "\n"
                for doc in dataset_fairseq
            ]
        )

    with open(
        os.path.join(
            output_folder,
            dataset_name,
            "{}.target".format(split_name),
        ),
        "w",
    ) as f:
        f.writelines(
            [
                doc[1].replace("\r", ">>").replace("\n", ">>") + "\n"
                for doc in dataset_fairseq
            ]
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_path",
        type=str,
        help="Path where to load the dataset(s)",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path where to save the converted dataset(s)",
    )
    parser.add_argument(
        "split_name",
        type=str,
        choices=["train", "dev"],
        help="Path where to save the converted dataset(s)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    args = parser.parse_args()
    
    logging.basicConfig(level=args.loglevel)

    datasets_filenames = (
        [os.path.join(args.input_path, fname) for fname in os.listdir(args.input_path)]
        if os.path.isdir(args.input_path)
        else [args.input_path]
    )

    for dataset_filename in datasets_filenames:

        logging.info("Loading {}".format(dataset_filename))
        with jsonlines.open(dataset_filename) as f:
            dataset = [e for e in f]

        source, target = convert_kilt_to_fairseq(
            dataset,
        )
        
        args.output, dataset_name, split_name
