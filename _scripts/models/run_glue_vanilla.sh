
# TODO: Make the tokenizer accessible at the end of the paper!

export PYTHONPATH=/Users/david/GoogleDrive/_MasterThesis # From the root folder of this repo
export GLUE_DIR=/Users/david/GoogleDrive/_MasterThesis/data/GLUE
export SAVEDIR=/Users/david/GoogleDrive/_MasterThesis/savedir

export PYTHONPATH=/home/david/_MasterThesis/ # From the root folder of this repo
export GLUE_DIR=/home/david/_MasterThesis/data/GLUE
export SAVEDIR=/home/david/_MasterThesis/savedir

# TODO: Make sure the parameters are adjusted to whatever task we have (if this is relevant...) !!!

# TODO: Implement multiple runs to have better experiment measurements

# BERT
MODEL_NAME=bert
MODEL_NAME_OR_PATH=bert-base-uncased

for TRY in 1 2 3
do
    for TASK in 'CoLA' 'MNLI' 'MNLI-MM' 'MRPC' 'SNLI' 'SST' 'SST-2' 'STS' 'STS-B' 'QQP' 'QNLI' 'RTE' 'WNLI'
    do

        python main.py \
          --model_type bert \
          --model_name_or_path bert-base-uncased \
          --task_name $TASK \
          --do_train \
          --do_eval \
          --do_lower_case \
          --data_dir $GLUE_DIR/$TASK/ \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 32 \
          --learning_rate 2e-5 \
          --num_train_epochs 3.0 \
          --output_dir $SAVEDIR/bert-$TRY-$TASK
    done done 2>&1 | tee $SAVEDIR/full_bert_glue.txt

for TRY in 1 2 3
do
    for TASK in 'CoLA' 'MNLI' 'MNLI-MM' 'MRPC' 'SNLI' 'SST' 'SST-2' 'STS' 'STS-B' 'QQP' 'QNLI' 'RTE' 'WNLI'
    do

        python main.py \
          --model_type albert \
          --model_name_or_path albert-base-v1 \
          --task_name $TASK \
          --do_train \
          --do_eval \
          --do_lower_case \
          --data_dir $GLUE_DIR/$TASK/ \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 32 \
          --learning_rate 2e-5 \
          --num_train_epochs 3.0 \
          --output_dir $SAVEDIR/albert-$TRY-$TASK
    done done 2>&1 | tee full_bert_glue.txt


python main.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir/Users/david/GoogleDrive/_MasterThesis/data/GLUE/MRPC/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /Users/david/GoogleDrive/_MasterThesis/savedir

--data_dir DATA_DIR
--model_type MODEL_TYPE
--model_name_or_path MODEL_NAME_OR_PATH
--task_name TASK_NAME
--output_dir OUTPUT_DIR