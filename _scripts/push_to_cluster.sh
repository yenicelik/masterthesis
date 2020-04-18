#!/usr/bin/env bash

# For Ubuntu 18.04, also add the lang specs to the .bashrc!
# export LC_CTYPE=en_US.UTF-8

#rsync -rPz -e 'ssh -p 2223' --exclude '*.png' --exclude 'data/' --exclude '*.tsv' --exclude 'venv/' --exclude 'analysis/' --exclude '.env' --progress /Users/david/GoogleDrive/_MasterThesis/ david@77.59.149.134:/home/david/_MasterThesis/
#rsync -rPz -e 'ssh -p 2223' --exclude '*.png' --exclude '*.tsv' --exclude 'venv/' --exclude 'analysis/' --exclude '.env' --progress /Users/david/GoogleDrive/_MasterThesis/ david@77.59.149.134:/home/david/_MasterThesis/
rsync -rPz -e 'ssh -p 2223' --exclude 'logs_BerniePoS/' --exclude 'savedir/' --exclude 'notebooks/2020_03_02 experiment with GLUE/_fine-tune-BERT/' --exclude '.git/' --exclude 'pictures/' --exclude 'data/' --exclude '_logdump/' --exclude '*.png' --exclude '*.tsv' --exclude 'venv/' --exclude 'analysis/' --exclude '.env' --progress /Users/david/GoogleDrive/_MasterThesis/ david@77.59.149.134:/home/david/_MasterThesis/

rsync -rPz --exclude 'logs_BerniePoS/' --exclude 'savedir/' --exclude 'notebooks/2020_03_02 experiment with GLUE/_fine-tune-BERT/' --exclude '.git/' --exclude 'pictures/' --exclude 'data/' --exclude '*.tsv' --exclude '_logdump/' --exclude '*.png' --exclude 'venv/' --exclude 'analysis/' --exclude '.env' --progress /Users/david/GoogleDrive/_MasterThesis/ yedavid@dalabgpu.inf.ethz.ch:/local/home/yedavid/_MasterThesis/
rsync -rPz --progress /Users/david/GoogleDrive/_MasterThesis/data/ yedavid@dalabgpu.inf.ethz.ch:/local/home/yedavid/_MasterThesis/data/

# Load modules for leonhard

# export PYTHONPATH="${PYTHONPATH}:~/"
# export PYTHONPATH=$PYTHONPATH:~/deeplearning/

# Push to leonhard
rsync -rPz --exclude 'notebooks/2020_03_02 experiment with GLUE/_fine-tune-BERT/' --exclude '.git/' --exclude 'pictures/' --exclude '_logdump/' --exclude '*.png' --exclude 'venv/' --exclude 'analysis/' --exclude '.env' --progress /Users/david/GoogleDrive/_MasterThesis/ david@login.leonhard.ethz.ch:/cluster/home/yedavid/_MasterThesis/

#bsub -I -R "rusage[mem=16000, ngpus_excl_p=1]" for TASK in 'CoLA'
#do
#
#    python notebooks/2020_03_08\ GLUE\ example\ training/main.py \
#      --model_type bernie \
#      --model_name_or_path bert-base-uncased \
#      --task_name CoLA \
#      --do_train \
#      --do_eval \
#      --do_lower_case \
#      --data_dir $GLUE_DIR/CoLA/ \
#      --max_seq_length 128 \
#      --per_gpu_train_batch_size 32 \
#      --learning_rate 2e-5 \
#      --num_train_epochs 3.0 \
#      --output_dir $SAVEDIR/bernie
#done
