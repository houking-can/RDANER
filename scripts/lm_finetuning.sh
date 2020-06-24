export TRAIN_FILE=/home/yhj/paper/ijcai-2020/daner/data/corpus/computer/train.txt
export TEST_FILE=/home/yhj/paper/ijcai-2020/daner/data/corpus/computer/test.txt
export TASK=lm_finetuning
export CUDA_VISIBLE_DEVICE=5
python run_lm_finetuning.py \
    --output_dir=output/$TASK \
    --model_type=bert \
    --model_name_or_path=/home/yhj/paper/ijcai-2020/daner/checkpoint/bert_base_cased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm