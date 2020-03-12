
# python notebooks/2020_01_04\ Clustering/try_all_clusters.py 2>&1 | tee


# Run on leonhard

export PYTHONPATH=/cluster/home/yedavid/_MasterThesis # From the root folder of this repo
export GLUE_DIR=/cluster/home/yedavid/_MasterThesis/data/GLUE
export SAVEDIR=/cluster/home/yedavid/_MasterThesis/savedir

for TASK in 'CoLA' 'MNLI' 'MNLI-MM' 'MRPC' 'SNLI' 'SST' 'SST-2' 'STS' 'STS-B' 'QQP' 'QNLI' 'RTE' 'WNLI'
do

    python otebooks/2020_03_08\ GLUE\ example\ training/main.py \
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
      --output_dir $SAVEDIR/bernie
done