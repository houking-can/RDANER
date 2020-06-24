# Run allennlp training locally


# edit these variables before running script

DATASET='chinese_clean'
TASK='text_classification'
with_finetuning='_finetune' #'_finetune'  # or '' for not fine tuning
dataset_size=1858

export BERT_VOCAB=/home/yhj/software/bert-base-chinese/vocab.txt
export BERT_WEIGHTS=/home/yhj/software/bert-base-chinese

export DATASET_SIZE=$dataset_size

CONFIG_FILE=allennlp_config/"$TASK""$with_finetuning".json

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=false
export TRAIN_PATH=data/$TASK/$DATASET/train.txt
export DEV_PATH=data/$TASK/$DATASET/dev.txt
export TEST_PATH=data/$TASK/$DATASET/test.txt

export CUDA_DEVICE=5

export GRAD_ACCUM_BATCH_SIZE=32
export NUM_EPOCHS=5
export LEARNING_RATE=0.00005

output_dir=/home/yhj/paper/ijcai-2020/daner/output
python -m allennlp.run train $CONFIG_FILE  --include-package scibert -s $output_dir