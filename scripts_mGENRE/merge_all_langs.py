import os

import numpy as np

from mgenre.utils import *
from tqdm.auto import tqdm

for seed, lang in enumerate(tqdm(sorted(our105_langs))):

    with open(os.path.join("/checkpoint/ndecao/wikipedia", lang, "titles_lang.source")) as f:
        source = list(tqdm(f, desc=lang))
    with open(os.path.join("/checkpoint/ndecao/wikipedia", lang, "titles_lang.target")) as f:
        target = list(tqdm(f, desc=lang))

    assert len(source) == len(target)

    np.random.seed(seed)
    np.random.shuffle(source)
    np.random.seed(seed)
    np.random.shuffle(target)

    with open(os.path.join("/checkpoint/ndecao/wikipedia/titles_lang_all/train.source"), "a") as f:
        f.writelines(source[:-1000])
    with open(os.path.join("/checkpoint/ndecao/wikipedia/titles_lang_all/train.target"), "a") as f:
        f.writelines(target[:-1000])

    with open(os.path.join("/checkpoint/ndecao/wikipedia/titles_lang_all/dev.source"), "a") as f:
        f.writelines(source[-1000:])
    with open(os.path.join("/checkpoint/ndecao/wikipedia/titles_lang_all/dev.target"), "a") as f:
        f.writelines(target[-1000:])

np.random.seed(42)
files_source= [open(
    "/checkpoint/ndecao/wikipedia/titles_lang_all_shards/shard{0}/train.spm.source".format(i), "w"
) for i in range(10)]
files_target = [open(
    "/checkpoint/ndecao/wikipedia/titles_lang_all_shards/shard{0}/train.spm.target".format(i), "w"
) for i in range(10)]

with open(os.path.join("/checkpoint/ndecao/wikipedia/titles_lang_all/train.spm.source")) as fs, open(
    os.path.join("/checkpoint/ndecao/wikipedia/titles_lang_all/train.spm.target")) as ft:
    for s, t in tqdm(zip(fs, ft), total=777105575):
        shard = np.random.choice(range(10))
        files_source[shard].write(s)
        files_target[shard].write(t)

for e in files_source + files_target:
    e.close()

# np.random.seed(42)
# files_target = [
#     open(
#         "/checkpoint/ndecao/wikipedia/lang_titles_all_shards/shard{0}/train.spm.target".format(
#             i
#         ),
#         "w",
#     )
#     for i in range(10)
# ]

# with open(
#     os.path.join("/checkpoint/ndecao/wikipedia/lang_titles_all/train.spm.target")
# ) as ft:
#     for t in tqdm(ft, total=777105575):
#         shard = np.random.choice(range(10))
#         files_target[shard].write(t)

# for e in files_target:
#     e.close()
