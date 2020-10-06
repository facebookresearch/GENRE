#!/bin/bash
python _private_scripts/convert_ed_to_kilt.py _private_data/ed/aida_testA.csv data/ed/aida-dev-kilt.jsonl _private_data/ed/persons.txt
python _private_scripts/convert_ed_to_kilt.py _private_data/ed/aida_testB.csv data/ed/aida-test-kilt.jsonl _private_data/ed/persons.txt
python _private_scripts/convert_ed_to_kilt.py _private_data/ed/aida_train.csv data/ed/aida-train-kilt.jsonl _private_data/ed/persons.txt
python _private_scripts/convert_ed_to_kilt.py _private_data/ed/wned-ace2004.csv data/ed/ace2004-test-kilt.jsonl _private_data/ed/persons.txt
python _private_scripts/convert_ed_to_kilt.py _private_data/ed/wned-aquaint.csv data/ed/aquaint-test-kilt.jsonl _private_data/ed/persons.txt
python _private_scripts/convert_ed_to_kilt.py _private_data/ed/wned-clueweb.csv data/ed/clueweb-test-kilt.jsonl _private_data/ed/persons.txt
python _private_scripts/convert_ed_to_kilt.py _private_data/ed/wned-msnbc.csv data/ed/msnbc-test-kilt.jsonl _private_data/ed/persons.txt
python _private_scripts/convert_ed_to_kilt.py _private_data/ed/wned-wikipedia.csv data/ed/wiki-test-kilt.jsonl _private_data/ed/persons.txt