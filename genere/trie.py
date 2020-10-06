from tqdm import tqdm
from collections import defaultdict


class Trie(object):
    def __init__(self, sequences, early_stop=False):
        if early_stop and len(sequences) == 1 and sequences[0] != []:
            sequences[0] = [2]

        next_sets = defaultdict(list)
        for seq in sequences:
            if len(seq) > 0:
                next_sets[seq[0]].append(seq[1:])

        self.early_stop = early_stop
        self._leaves = {k: Trie(v, early_stop=early_stop) for k, v in next_sets.items()}

    def get(self, indices, sep=None, callback=None):
        if len(indices) == 0:
            if sep is not None and callback is not None and sep in self._leaves.keys():
                subtrie = callback()
                if subtrie is not None:
                    for k, v in subtrie._leaves.items():
                        self._leaves[k] = v
                    del self._leaves[sep]

            return list(self._leaves.keys())
        elif indices[0] not in self._leaves:
            return []
        else:
            return self._leaves[indices[0]].get(indices[1:], sep=sep, callback=callback)
