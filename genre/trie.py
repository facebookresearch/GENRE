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


class DummyTrieMention(object):
    def __init__(self, return_values):
        self._return_values = return_values

    def get(self, indices=None):
        return self._return_values


class DummyTrieEntity(object):
    def __init__(self, return_values, codes):
        self._return_values = list(
            set(return_values).difference(set(codes[e] for e in "{}["))
        )
        self._codes = codes

    def get(self, indices, depth=0):
        if len(indices) == 0 and depth == 0:
            return self._codes["}"]
        elif len(indices) == 0 and depth == 1:
            return self._codes["["]
        elif len(indices) == 0:
            return self._return_values
        elif len(indices) == 1 and indices[0] == self._codes["]"]:
            return self._codes["EOS"]
        else:
            return self.get(indices[1:], depth=depth + 1)
