
export PYTHONPATH=$(pwd) # From the root folder of this repo

# cd into the notebooks file ...

python main.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --output_dir $SAVEDIR