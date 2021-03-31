#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


mkdir wikinews
cd wikinews

for LANG in ar bg bs ca cs de el en eo es fa fi fr he hu it ja ko nl no pl pt ro ru sd sq sr sv ta th tr uk zh
do
    wget http://wikipedia.c3sl.ufpr.br/${LANG}wikinews/20191001/${LANG}wikinews-20191001-pages-articles-multistream.xml.bz2
done

for LANG in ar bg bs ca cs de el en eo es fa fi fr he hu it ja ko nl no pl pt ro ru sd sq sr sv ta th tr uk zh
do
    wikiextractor ${LANG}wikinews-20191001-pages-articles-multistream.xml.bz2 -o ${LANG} --links --section_hierarchy --lists --sections
done
