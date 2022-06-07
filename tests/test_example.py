import pickle
from unittest.mock import ANY

import pytest

from genre.fairseq_model import GENRE, GENREHubInterface
from genre.trie import Trie


@pytest.fixture(scope="session")
def kilt_trie():
    # load the prefix tree (trie)
    with open("./data/kilt_titles_trie_dict.pkl", "rb") as f:
        trie = Trie.load_from_dict(pickle.load(f))
    return trie


@pytest.fixture(scope="session")
def fairseq_wikipage_retrieval():
    model = GENRE.from_pretrained("./models/fairseq_wikipage_retrieval").eval()
    return model


EXPECTED_RESULTS_DOCUMENT_RETRIEVAL = [
    [
        {"text": "Albert Einstein", "score": ANY},
        {"text": "Werner Bruschke", "score": ANY},
        {"text": "Werner von Habsburg", "score": ANY},
        {"text": "Werner von Moltke", "score": ANY},
        {"text": "Werner von Eichstedt", "score": ANY},
    ]
]


def test_example_document_retrieval(
    kilt_trie: Trie, fairseq_wikipage_retrieval: GENREHubInterface
):
    sentences = ["Einstein was a German physicist."]
    results = fairseq_wikipage_retrieval.sample(
        sentences,
        prefix_allowed_tokens_fn=lambda batch_id, sent: kilt_trie.get(sent.tolist()),
    )
    assert results == EXPECTED_RESULTS_DOCUMENT_RETRIEVAL
