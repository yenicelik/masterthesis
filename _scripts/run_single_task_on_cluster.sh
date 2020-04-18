export PYTHONPATH=/Users/david/GoogleDrive/_MasterThesis # $(pwd)
export GLUE_DIR=/Users/david/GoogleDrive/_MasterThesis/data/GLUE
export SAVEDIR=/Users/david/GoogleDrive/_MasterThesis/savedir

# All tasks which have low resources!
export PYTHONPATH=/home/david/_MasterThesis # From the root folder of this repo
export GLUE_DIR=/home/david/_MasterThesis/data/GLUE
export SAVEDIR=/home/david/_MasterThesis/savedir

# dalab gpus
export PYTHONPATH=/local/home/yedavid/_MasterThesis # From the root folder of this repo
export GLUE_DIR=/local/home/yedavid/_MasterThesis/data/GLUE
export SAVEDIR=/local/home/yedavid/_MasterThesis/savedir

#    echo bernie;\
#    echo $TASK; \
#    python notebooks/2020_03_08\ GLUE\ example\ training/main.py \
#      --model_type bernie \
#      --model_name_or_path bert-base-uncased \
#      --task_name $TASK \
#      --do_train \
#      --do_eval \
#      --do_lower_case \
#      --data_dir $GLUE_DIR/$TASK/ \
#      --max_seq_length 128 \
#      --per_gpu_train_batch_size 32 \
#      --learning_rate 2e-5 \
#      --num_train_epochs 3.0 \
#      --overwrite_cache \
#      --output_dir $SAVEDIR/bernie-$TASK; \

#    echo albert;\
#    echo $TASK; \
#    python notebooks/2020_03_08\ GLUE\ example\ training/main.py \
#      --model_type albert \
#      --model_name_or_path albert-base-v1 \
#      --task_name $TASK \
#      --do_train \
#      --do_eval \
#      --do_lower_case \
#      --data_dir $GLUE_DIR/$TASK/ \
#      --max_seq_length 128 \
#      --per_gpu_train_batch_size 32 \
#      --learning_rate 2e-5 \
#      --num_train_epochs 3.0 \
#      --overwrite_cache \
#      --output_dir $SAVEDIR/albert-$TASK;

for TASK in 'MNLI' 'SNLI' 'QQP' # 'CoLA' 'MRPC' 'SST' 'SST-2' 'STS' 'STS-B' 'QNLI' 'RTE' 'WNLI'
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
      --overwrite_cache \
      --overwrite_output_dir \
      --seed 101 \
      --output_dir $SAVEDIR/bert-$TASK
done 2>&1 | tee $SAVEDIR/big_task_bert_models_glue-101.txt

#    echo bernie-pos;\
#    echo $TASK; \
#    echo $PYTHONPATH; \
#    echo $GLUE_DIR; \
#    echo $SAVEDIR; \

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
      --output_dir $SAVEDIR/bernie_pos-$TASK-20200319_s101;
done 2>&1 | tee $SAVEDIR/sst2_bernie_pos_models_glue_20200319_s101.txt

# Dis one
#       --device 0 \
#       --no_cuda
export CUDA_LAUNCH_BLOCKING=1
CUDA_LAUNCH_BLOCKING=1
# for TASK in 'CoLA' 'MRPC' 'SST-2' 'STS-B' 'QNLI' 'RTE' 'WNLI'
# for TASK in  'WNLI' 'RTE' 'QNLI' # 'STS-B' 'SST-2' 'MRPC' 'CoLA' #
#for TASK in  'MNLI' 'SNLI' 'QQP' # 'STS-B' 'SST-2' 'MRPC' 'CoLA' #
for TASK in 'WNLI' 'RTE' 'QNLI' 'STS-B'   # 'CoLA' 'MRPC' 'SST-2'
do
    echo bernie-meaning;\
    echo $TASK; \
    python notebooks/2020_03_08\ GLUE\ example\ training/main.py \
      --model_type bernie_meaning \
      --model_name_or_path bert-base-uncased \
      --task_name $TASK \
      --do_train \
      --additional_pretraining True \
      --do_eval \
      --do_lower_case \
      --data_dir $GLUE_DIR/$TASK/ \
      --max_seq_length 128 \
      --per_gpu_train_batch_size 32 \
      --learning_rate 2e-5 \
      --num_train_epochs 3.0 \
      --overwrite_output_dir \
      --overwrite_cache \
      --seed 42 \
      --only_finetune_newly_added_embeddings True \
      --output_meaning_dir $SAVEDIR/bernie_meaning_cache \
      --output_dir $SAVEDIR/bernie_meaning_20200405_finetuned_42;
done 2>&1 | tee $SAVEDIR/bernie_meaning_20200413_finetuned_42.txt

# JUST SOME TEST
for TASK in 'CoLA' # 'MRPC' 'SST-2' 'STS-B' 'QNLI' 'RTE' 'WNLI'
do
    echo bert;\
    echo $TASK; \
    python notebooks/2020_03_08\ GLUE\ example\ training/main.py \
      --model_type bert \
      --model_name_or_path bert-base-uncased \
      --task_name $TASK \
      --do_train \
      --additional_pretraining True \
      --do_eval \
      --do_lower_case \
      --data_dir $GLUE_DIR/$TASK/ \
      --max_seq_length 128 \
      --per_gpu_train_batch_size 32 \
      --learning_rate 2e-5 \
      --num_train_epochs 1.0 \
      --overwrite_output_dir \
      --overwrite_cache \
      --seed 101 \
      --output_meaning_dir $SAVEDIR/bernie_meaning_cache \
      --output_dir $SAVEDIR/bernie_meaning_20200404;
done 2>&1 | tee $SAVEDIR/bernie_meaning_20200404.txt

for TASK in 'CoLA' 'MRPC' 'SST-2' 'STS-B' 'QNLI' 'RTE' 'WNLI'
do
    echo bert; \
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
      --output_dir $SAVEDIR/bert-$TASK-20200316;
done 2>&1 | tee $SAVEDIR/small_task_bert_models_glue_20200316.txt


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
