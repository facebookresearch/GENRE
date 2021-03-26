#!/bin/bash

# rm *0.pt *1.pt *2.pt *3.pt *4.pt *5.pt *6.pt *7.pt *8.pt *9.pt

# remember to check `restore_file` and `total_num_udpates` in run_bart_slurm.py

cd src/fairseq

python ../../scripts/run_bart_slurm.py \
    -n 48 \
    -g 8 \
    -t 1 \
    -p "titles_lang_all_for_TR2016" \
    -d "/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_for_TR2016_shards_bins/shard0:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_for_TR2016_shards_bins/shard1:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_for_TR2016_shards_bins/shard2:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_for_TR2016_shards_bins/shard3:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_for_TR2016_shards_bins/shard4:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_for_TR2016_shards_bins/shard5:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_for_TR2016_shards_bins/shard6:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_for_TR2016_shards_bins/shard7:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_for_TR2016_shards_bins/shard8:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_for_TR2016_shards_bins/shard9" \
    --partition ael \
    --time 7200 \
    --resume-failed


# python ../mGENRE/scripts/run_bart_slurm.py \
#     -n 32 \
#     -g 8 \
#     -t 1 \
#     -p "titles_lang_all" \
#     -d "/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard0:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard1:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard2:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard3:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard4:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard5:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard6:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard7:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard8:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard9" \
#     --partition ael \
#     --resume-failed


# python ../../scripts/run_bart_slurm.py \
#     -n 1 \
#     -g 8 \
#     -t 1 \
#     -p "titles_lang_all_fine_tune_TACKBP2015_noNIL" \
#     -d "/checkpoint/fabiopetroni/mGENRE/TACKBP2015_noNIL/bin" \
#     --partition ael \
#     --resume-failed


# python ../../scripts/run_bart_slurm.py \
#     -n 32 \
#     -g 8 \
#     -t 1 \
#     -p "titles_lang_all_1.2B" \
#     -d "/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard0:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard1:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard2:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard3:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard4:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard5:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard6:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard7:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard8:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard9" \
#     --partition ael \
#     --time 14400 \
#     --resume-failed

#
#    


# python ../mGENRE/scripts/run_bart_slurm.py \
#     -n 32 \
#     -g 8 \
#     -t 1 \
#     -p "titles_lang_all" \
#     -d "/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard0:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard1:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard2:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard3:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard4:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard5:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard6:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard7:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard8:/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_all_shards_bins/shard9" \
#     --partition ael \
#     --resume-failed


# python ../mGENRE/scripts/run_bart_slurm.py \
#     -n 8 \
#     -g 8 \
#     -t 1 \
#     -p "canonical_title_abstract_mewsli_1M" \
#     -d "/checkpoint/fabiopetroni/mGENRE/wikipedia/canonical_title_abstract_mewsli_1M/bin" \
#     --constraint volta32gb \
#     --mem 400G \
#     --resume-failed
    
# python ../mGENRE/scripts/run_bart_slurm.py \
#     -n 8 \
#     -g 8 \
#     -t 1 \
#     -p "titles_lang_abstract_target_switching_mewsli_1M" \
#     -d "/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_abstract_target_switching_mewsli_1M/bin" \
#     --constraint volta32gb \
#     --mem 400G \
#     --resume-failed
    
# python ../mGENRE/scripts/run_bart_slurm.py \
#     -n 8 \
#     -g 8 \
#     -t 1 \
#     -p "titles_lang_abstract_mewsli_1M" \
#     -d "/checkpoint/fabiopetroni/mGENRE/wikipedia/titles_lang_abstract_mewsli_1M/bin" \
#     --constraint volta32gb \
#     --mem 400G \
#     --resume-failed
    
# python ../mGENRE/scripts/run_bart_slurm.py \
#     -n 8 \
#     -g 8 \
#     -t 1 \
#     -p "lang_titles_abstract_target_switching_mewsli_1M" \
#     -d "/checkpoint/fabiopetroni/mGENRE/wikipedia/lang_titles_abstract_target_switching_mewsli_1M/bin" \
#     --constraint volta32gb \
#     --mem 400G \
#     --resume-failed
    
# python ../mGENRE/scripts/run_bart_slurm.py \
#     -n 8 \
#     -g 8 \
#     -t 1 \
#     -p "lang_titles_abstract_mewsli_1M" \
#     -d "/checkpoint/fabiopetroni/mGENRE/wikipedia/lang_titles_abstract_mewsli_1M/bin" \
#     --constraint volta32gb \
#     --mem 400G \
#     --resume-failed

# python ../mGENRE/scripts/run_bart_slurm.py \
#     -n 8 \
#     -g 8 \
#     -t 1 \
#     -p "marginal_abstract_mewsli_1M" \
#     -d "/checkpoint/fabiopetroni/mGENRE/wikipedia/marginal_abstract_mewsli_1M/bin" \
#     --constraint volta32gb \
#     --mem 400G \
#     --resume-failed
    
    
# python ../mGENRE/scripts/run_bart_slurm.py \
#     -n 1 \
#     -g 1 \
#     -t 1 \
#     -p "test_marginal_server" \
#     -d "/checkpoint/fabiopetroni/mGENRE/wikipedia/test_marginal/bin" \
#     --constraint volta32gb \
#     --mem 400G \
#     --resume-failed
