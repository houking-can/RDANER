# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/yhj/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/yhj/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/yhj/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/yhj/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate torch1.3

DATASET=
TASK=
with_finetuning='_finetune'
dataset_size=
export NUM_EPOCHS=
export CUDA_DEVICE=
SEED=

export BERT_VOCAB=
export BERT_WEIGHTS=

export TRAIN_PATH=
export DEV_PATH=
export TEST_PATH=
output_dir=

export DATASET_SIZE=$dataset_size

CONFIG_FILE=allennlp_config/"$TASK""$with_finetuning".json


PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`

export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=false

export GRAD_ACCUM_BATCH_SIZE=32
export LEARNING_RATE=0.00002

python -m allennlp.run train $CONFIG_FILE  --include-package scibert -s $output_dir