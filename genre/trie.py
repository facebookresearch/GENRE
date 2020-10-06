from tqdm import tqdm
from collections import defaultdict


class Trie(object):
    def __init__(self, sequences):
        next_sets = defaultdict(list)
        for seq in sequences:
            if len(seq) > 0:
                next_sets[seq[0]].append(seq[1:])

        self._leaves = {k: Trie(v) for k, v in next_sets.items()}

    def get(self, indices):
        if len(indices) == 0:
            return list(self._leaves.keys())
        elif indices[0] not in self._leaves:
            return []
        else:
            return self._leaves[indices[0]].get(indices[1:])
