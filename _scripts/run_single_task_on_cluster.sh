export PYTHONPATH=/Users/david/GoogleDrive/_MasterThesis # $(pwd)
export GLUE_DIR=/Users/david/GoogleDrive/_MasterThesis/data/GLUE
export SAVEDIR=/Users/david/GoogleDrive/_MasterThesis/savedir/

# All tasks which have low resources!
export PYTHONPATH=/home/david/_MasterThesis # From the root folder of this repo
export GLUE_DIR=/home/david/_MasterThesis/data/GLUE
export SAVEDIR=/home/david/_MasterThesis/savedir

for TASK in 'CoLA' 'MRPC' 'SST' 'SST-2' 'STS' 'STS-B' 'QNLI' 'RTE' 'WNLI'
do
    echo bernie;\
    echo $TASK; \
    python notebooks/2020_03_08\ GLUE\ example\ training/main.py \
      --model_type bernie \
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
      --overwrite_cache \
      --output_dir $SAVEDIR/bernie-$TASK; \
    echo bert;\
    echo $TASK; \
    python notebooks/2020_03_08\ GLUE\ example\ training/main.py \
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
      --overwrite_cache \
      --output_dir $SAVEDIR/bert-$TASK; \
    echo albert;\
    echo $TASK; \
    python notebooks/2020_03_08\ GLUE\ example\ training/main.py \
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
      --overwrite_cache \
      --output_dir $SAVEDIR/albert-$TASK;
done 2>&1 | tee $SAVEDIR/small_task_all_models_glue.txt

for TASK in 'CoLA' 'MRPC' 'SST-2' 'STS-B' 'QNLI' 'RTE' 'WNLI'
do
    echo bernie-pos;\
    echo $TASK; \
    python notebooks/2020_03_08\ GLUE\ example\ training/main.py \
      --model_type bernie_pos \
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
      --overwrite_output_dir \
      --overwrite_cache \
      --seed 101 \
      --output_dir $SAVEDIR/bernie_pos-$TASK-20200315;
done 2>&1 | tee $SAVEDIR/small_task_bernie_pos_models_glue_20200315.txt


for TASK in 'CoLA' # 'MRPC' 'SST-2' 'STS-B' 'QNLI' 'RTE' 'WNLI'
do
    echo bernie-meaning;\
    echo $TASK; \
    python notebooks/2020_03_08\ GLUE\ example\ training/main.py \
      --model_type bernie_meaning \
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
      --overwrite_output_dir \
      --overwrite_cache \
      --seed 101 \
      --output_dir $SAVEDIR/bernie_meaning-$TASK-20200315;
done 2>&1 | tee $SAVEDIR/small_task_bernie_meaning_models_glue_20200315.txt

for TASK in 'CoLA' 'MRPC' 'SST-2' 'STS-B' 'QNLI' 'RTE' 'WNLI'
do
    echo bert;\
    echo $TASK; \
    python notebooks/2020_03_08\ GLUE\ example\ training/main.py \
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
      --overwrite_output_dir \
      --overwrite_cache \
      --seed 101 \
      --output_dir $SAVEDIR/bert-$TASK-20200314;
done 2>&1 | tee $SAVEDIR/small_task_bert_models_glue_20200314.txt


for TASK in 'CoLA' 'MRPC' 'SST-2' 'STS-B' 'QNLI' 'RTE' 'WNLI'
do
    echo albert;\
    echo $TASK; \
    python notebooks/2020_03_08\ GLUE\ example\ training/main.py \
      --model_type albert \
      --model_name_or_path albert-base-v2 \
      --task_name $TASK \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir $GLUE_DIR/$TASK/ \
      --max_seq_length 128 \
      --per_gpu_train_batch_size 32 \
      --learning_rate 2e-5 \
      --num_train_epochs 3.0 \
      --overwrite_output_dir \
      --overwrite_cache \
      --seed 101 \
      --output_dir $SAVEDIR/albert-$TASK-20200314;
done 2>&1 | tee $SAVEDIR/small_task_albert_models_glue_20200314.txt
