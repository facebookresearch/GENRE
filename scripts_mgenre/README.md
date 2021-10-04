Summary of the scripts:
* `download_news.sh`: donwloads all Wikinews
* `download_wiki.sh`: donwloads all Wikipedia
* `evaluate_kilt_dataset.py`: evaluates a model on a KILT style dataset and produces output files (used for all datasets since we convert them in a KILT style `jsonl`)
* `evaluate_mel.py`: evaluates accuracy/F1/precision/recall of KILT style dataset (gold and prediction)
* `preprocess_TR2016.py`: takes TR2016 and it solves the hyperlinks to Wikidata IDs and it transforms it to KILT format
* `preprocess_anchors.py`: solves hyperlinks to Wikidata IDs and saves the Wikipedia file again
* `preprocess_extract.py`: extract Wikipedia files from wikiextractor and it constructs a dictionary (later saved into a pickle file)
* `preprocess_fairseq.sh`: tokenizes and binarizes a dataset given a path
* `preprocess_mention_dicts.py`: generates mention tables
* `preprocess_mewsli.py`: takes mewsli-9 as preprocessed from http://goo.gle/mewsli-dataset and it solves the hyperlinks to Wikidata IDs and it transforms it to KILT format
* `preprocess_mgenre.py`: takes Wikipedia a pickle file (with solved hyperlinks) and it generates training data for the seq2seq objectives
* `preprocess_sentencepiece.py`: Mikel stript for parallel fast tokenization (called from `preprocess_fairseq.sh`)
* `preprocess_tries.py`: it generates prefix tree dictionaries for constrained generation
* `preprocess_wikidata.py`: preprocess Wikidata 1) reducing the size from 1TB to 25GB removing unused items 2) generates useful dictionaries title -> ID, ID -> title or alias tables
* `preprocess_wikinews.py`: preprocess Wikinews into KILT style files and it splits into train/dev/test
