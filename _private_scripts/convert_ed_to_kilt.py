import os
import argparse
import jsonlines
from tqdm import tqdm


# code from https://github.com/lephong/mulrel-nel
def read_csv_file(path):
    data = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            comps = line.strip().split("\t")
            doc_name = comps[0] + " " + comps[1]
            mention = comps[2]
            lctx = comps[3]
            rctx = comps[4]

            if comps[6] != "EMPTYCAND":
                cands = [c.split(",") for c in comps[6:-2]]
                cands = [
                    (",".join(c[2:]).replace('"', "%22").replace(" ", "_"), float(c[1]))
                    for c in cands
                ]
            else:
                cands = []

            gold = comps[-1].split(",")
            if gold[0] == "-1":
                gold = (
                    ",".join(gold[2:]).replace('"', "%22").replace(" ", "_"),
                    1e-5,
                    -1,
                )
            else:
                gold = (
                    ",".join(gold[3:]).replace('"', "%22").replace(" ", "_"),
                    1e-5,
                    -1,
                )

            if doc_name not in data:
                data[doc_name] = []
            data[doc_name].append(
                {
                    "mention": mention,
                    "context": (lctx, rctx),
                    "candidates": cands,
                    "gold": gold,
                }
            )
    return data


def load_person_names(path):
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            data.append(line.strip().replace(" ", "_"))
    return set(data)


def find_coref(ment, mentlist, person_names):
    cur_m = ment["mention"].lower()
    coref = []
    for m in mentlist:
        if len(m["candidates"]) == 0 or m["candidates"][0][0] not in person_names:
            continue

        mention = m["mention"].lower()
        start_pos = mention.find(cur_m)
        if start_pos == -1 or mention == cur_m:
            continue

        end_pos = start_pos + len(cur_m) - 1
        if (start_pos == 0 or mention[start_pos - 1] == " ") and (
            end_pos == len(mention) - 1 or mention[end_pos + 1] == " "
        ):
            coref.append(m)

    return coref


def with_coref(dataset, person_names):
    for data_name, content in dataset.items():
        for cur_m in content:
            coref = find_coref(cur_m, content, person_names)
            if coref is not None and len(coref) > 0:
                cur_cands = {}
                for m in coref:
                    for c, p in m["candidates"]:
                        cur_cands[c] = cur_cands.get(c, 0) + p
                for c in cur_cands.keys():
                    cur_cands[c] /= len(coref)
                cur_m["candidates"] = sorted(
                    list(cur_cands.items()), key=lambda x: x[1]
                )[::-1]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        type=str,
        help="input file",
    )

    parser.add_argument(
        "output",
        type=str,
        help="output file",
    )

    parser.add_argument(
        "persons",
        type=str,
        help="persons.txt file",
    )

    args = parser.parse_args()

    dataset = read_csv_file(args.input)
    person_names = load_person_names(args.persons)
    with_coref(dataset, person_names)
    dataset = [f for e in dataset.values() for f in e]

    dataset_jsonl = []
    for id_, doc in enumerate(tqdm(dataset, desc="Converting")):

        if doc["gold"][0] == "Lujaizui":
            doc["gold"][0] = "Lujiazui"

        doc["context"] = [e.replace("EMPTYCTXT", "") for e in doc["context"]]
        dataset_jsonl.append(
            {
                "id": id_,
                "input": "{} [START_ENT] {} [END_ENT] {}".format(
                    doc["context"][0], doc["mention"], doc["context"][1]
                ).strip(),
                "output": [{"answer": doc["gold"][0].replace("_", " "), "provenance": [{
                    "title": doc["gold"][0].replace("_", " ")
                }]}],
                "meta": {
                    "left_context": doc["context"][0],
                    "right_context": doc["context"][1],
                    "mention": doc["mention"],
                },
                "candidates": [e[0].replace("_", " ") for e in doc["candidates"]],
            }
        )

    with jsonlines.open(args.output, "w") as f:
        f.write_all(dataset_jsonl)
