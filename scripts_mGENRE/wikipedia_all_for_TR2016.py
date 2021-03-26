import os
import pickle
import re

import numpy as np

import jsonlines
from tqdm.auto import tqdm

with open(
    "/checkpoint/fabiopetroni/mGENRE/wikidata/lang_title2wikidataID-normalized.pkl",
    "rb",
) as f:
    lang_title2wikidataID = pickle.load(f)

data = []
for fname in os.listdir("/checkpoint/fabiopetroni/mGENRE/TR2016"):
    if "test" in fname:
        with jsonlines.open(
            os.path.join("/checkpoint/fabiopetroni/mGENRE/TR2016", fname)
        ) as f:
            data += list(f)


no_mentions = {
    (d["meta"]["mention"], wikidataID)
    for d in data
    for wikidataID in d["output"][0]["answer"]
}

np.random.seed(42)
files_source = [
    open(
        "/checkpoint/ndecao/wikipedia/titles_lang_all_for_TR2016_shards_v2/shard{0}/train.spm.source".format(
            i
        ),
        "w",
    )
    for i in range(10)
]
files_target = [
    open(
        "/checkpoint/ndecao/wikipedia/titles_lang_all_for_TR2016_shards_v2/shard{0}/train.spm.target".format(
            i
        ),
        "w",
    )
    for i in range(10)
]

with open(
    "/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all/train.source"
) as src_s, open(
    "/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all/train.target"
) as src_t, open(
    "/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all/train.spm.source"
) as src_ss, open(
    "/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all/train.spm.target"
) as src_ts:
    for s_, t_, ss_, ts_ in zip(tqdm(src_s, total=777105575), src_t, src_ss, src_ts):
        mention_title = (
            re.findall(r"\[START\] (.*) \[END\]", s_)[0],
            list(lang_title2wikidataID[tuple(reversed(t_.strip().split(" >> ")))])[0],
        )
        if mention_title not in no_mentions:
            shard = np.random.choice(range(10))
            files_source[shard].write(ss_)
            files_target[shard].write(ts_)
