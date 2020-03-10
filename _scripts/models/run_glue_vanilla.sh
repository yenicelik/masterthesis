
# TODO: Make the tokenizer accessible at the end of the paper!

export PYTHONPATH=$(pwd) # From the root folder of this repo
export GLUE_DIR=/Users/david/GoogleDrive/_MasterThesis/data/GLUE
export SAVEDIR=/Users/david/GoogleDrive/_MasterThesis/savedir/

# TODO: Make sure the parameters are adjusted to whatever task we have (if this is relevant...) !!!

for TASK in 'COLA', 'MNLI', 'MNLI-MM', 'MRPC', 'SST-2', 'STS-B', 'QQP', 'QNLI', 'RTE', 'WNLI'
do

    python main.py \
      --model_type bert \
      --model_name_or_path bert-base-uncased \
      --task_name MRPC \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir $GLUE_DIR/$TASK/ \
      --max_seq_length 128 \
      --per_gpu_train_batch_size 32 \
      --learning_rate 2e-5 \
      --num_train_epochs 1.0 \
      --output_dir $SAVEDIR

done