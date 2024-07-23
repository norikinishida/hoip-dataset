import argparse
import csv
# import os

from tqdm import tqdm

import utils


def main(args):
    assert args.path.endswith(".train.json")

    train_examples = utils.read_json(args.path)
    dev_examples = utils.read_json(args.path.replace(".train", ".dev"))
    test_examples = utils.read_json(args.path.replace(".train", ".test"))
    splits = {
        "train": train_examples,
        "dev": dev_examples,
        "test": test_examples
    }

    with open(args.path.replace(".train.json", ".csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_key", "split", "pubmed_id", "sentences", "head_entity_id", "head_entity_name", "fg_head_entity_id", "fg_head_entity_name", "relation_id", "relation_name", "tail_entity_id", "tail_entity_name", "fg_tail_entity_id", "fg_tail_entity_name", "context_id", "context_name", "course_id", "course_name", "merged"])
        for split, examples in splits.items():
            for example in tqdm(examples):
                doc_key = example["doc_key"]
                pubmed_id = example["pubmed_id"]
                sentences = " \n".join(example["sentences"])
                for triple in example["triples"]:
                    head_entity_id = triple["head_entity"]["id"]
                    head_entity_name = triple["head_entity"]["name"]
                    fg_head_entity_id = triple["fg_head_entity"]["id"]
                    fg_head_entity_name = triple["fg_head_entity"]["name"]
                    relation_id = triple["relation"]["id"]
                    relation_name = triple["relation"]["name"]
                    tail_entity_id = triple["tail_entity"]["id"]
                    tail_entity_name = triple["tail_entity"]["name"]
                    fg_tail_entity_id = triple["fg_tail_entity"]["id"]
                    fg_tail_entity_name = triple["fg_tail_entity"]["name"]
                    context_id = triple["context"]["id"]
                    context_name = triple["context"]["name"]
                    course_id = triple["course"]["id"]
                    course_name = triple["course"]["name"]
                    merged = triple["merged"]
                    writer.writerow([doc_key, split, pubmed_id, sentences, head_entity_id, head_entity_name, fg_head_entity_id, fg_head_entity_name, relation_id, relation_name, tail_entity_id, tail_entity_name, fg_tail_entity_id, fg_tail_entity_name, context_id, context_name, course_id, course_name, merged])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    main(args=args)
