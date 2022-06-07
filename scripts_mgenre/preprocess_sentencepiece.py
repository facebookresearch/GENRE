# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import random
import sys
from collections import Counter
from multiprocessing import Pool

import sentencepiece as spm


class MultiprocessingEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        global sp, old2new
        sp = spm.SentencePieceProcessor(model_file=self.args.model)
        old2new = None
        if self.args.product_vocab_size is not None:
            assert sp.vocab_size() <= self.args.product_vocab_size**2
            rand = random.Random(self.args.seed)
            old2new = [
                (x // self.args.product_vocab_size, x % self.args.product_vocab_size)
                for x in rand.sample(
                    range(self.args.product_vocab_size**2), sp.vocab_size()
                )
            ]

    def encode(self, line):
        global sp, old2new
        ids = sp.encode_as_pieces(line)
        if old2new:
            ids = [x for old in ids for x in old2new[old]]
        if self.args.offset > 0:
            ids = [x + self.args.offset for x in ids]
        return list(map(str, ids))

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="SentencePiece model")
    parser.add_argument(
        "--product-vocab-size",
        type=int,
        help="Product vocabulary size (disabled by default)",
    )
    parser.add_argument(
        "--seed", type=int, default=13, help="The seed for the product vocabulary"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="The offset to add to every output id (defaults to 0)",
    )
    parser.add_argument(
        "-i", "--inputs", nargs="+", default=["-"], help="input files to filter/encode"
    )
    parser.add_argument(
        "-o", "--outputs", nargs="+", default=["-"], help="path to save encoded outputs"
    )
    parser.add_argument("--keep-empty", action="store_true", help="keep empty lines")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(
        args.outputs
    ), "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-"
            else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-"
            else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)
