import argparse
import pickle

import jsonlines
from tqdm.auto import tqdm, trange

from genre import GENRE


def batch_iter(obj, batch_size=1):
    out = []
    for item in obj:
        if len(out) == batch_size:
            yield out
            out = []
        out.append(item)

    if len(out):
        yield out


parser = argparse.ArgumentParser()

parser.add_argument(
    "--rank",
    type=int,
)

parser.add_argument(
    "--first_half",
    action="store_true",
)

parser.add_argument(
    "--second_half",
    action="store_true",
)


args = parser.parse_args()


# loading model
context2answer = (
    GENRE.from_pretrained(
        "/checkpoint/ndecao/2020-11-03/nq_context2answer.bart_large.ls0.1.mt2048.uf4.mu20000.dr0.1.atdr0.1.actdr0.0.wd0.01.adam.beta9999.eps1e-08.clip0.1.lr3e-05.warm500.fp16.ngpu8",
        "checkpoint43.pt",
    )
    .eval()
    .to("cuda:{}".format(args.rank % 8))
)

answer_context2query = (
    GENRE.from_pretrained(
        "/checkpoint/ndecao/2020-11-03/nq_answer_context2query.bart_large.ls0.1.mt2048.uf4.mu20000.dr0.1.atdr0.1.actdr0.0.wd0.01.adam.beta9999.eps1e-08.clip0.1.lr3e-05.warm500.fp16.ngpu8",
        "checkpoint43.pt",
    )
    .eval()
    .to("cuda:{}".format(args.rank % 8))
)

batch_size = 48
data = []
with jsonlines.open(
    "/checkpoint/fabiopetroni/GENRE/checkpoint/GeNeRe/data/kilt/kilt_{}.jsonl".format(
        args.rank
    )
) as f:
    inputs = [e for e in tqdm(batch_iter(f, batch_size))]

if args.first_half:
    inputs = inputs[: len(inputs) // 2]
elif args.second_half:
    inputs = inputs[len(inputs) // 2 :]

# if args.first_half:
#     inputs = inputs[:3]
# elif args.second_half:
#     inputs = inputs[3:]

iter_ = tqdm(inputs, smoothing=0)
for psgs in iter_:

    psgs = [
        psg["text"]
        for psg in psgs
        if all(
            e not in psg["section"].lower()
            for e in (
                "see also",
                "references",
                "external link",
                "further reading",
                "notes",
            )
        )
    ]
    if psgs:
        outputs_context2answer = context2answer.sample(psgs, beams=5)
        for ans_psgs in batch_iter(
            [
                "{} >> {}".format(answer["text"], psg)
                for answers, psg in zip(outputs_context2answer, psgs)
                for answer in answers
            ],
            batch_size,
        ):
            outputs_context2query = [
                e[0]["text"] for e in answer_context2query.sample(ans_psgs)
            ]
            for q, ac in zip(outputs_context2query, ans_psgs):
                data += [[q] + ac.split(" >> ")]
    iter_.set_postfix(data=len(data))

with open(
    "qac_{}_{}_{}.pkl".format(args.rank, args.first_half, args.second_half), "wb"
) as f:
    pickle.dump(data, f)
