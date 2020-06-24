import os
import re
import shutil
import json

HOME = '/home/yhj/paper/ijcai-2020/daner'
dataset_info = f'{HOME}/data/dataset_info.json'
dataset_info = json.load(open(dataset_info))['ner']

script_base = f'{HOME}/scripts/'
model_base = f'{HOME}/checkpoint'
data_base = f'{HOME}/data/split/ner'


def run_lm(model, dataset, ratio, seed, device):
    print(model, dataset, ratio, seed, device)
    base_ner_output = f'{HOME}/output/lm/{dataset}/{ratio}'
    if os.path.exists(base_ner_output):
        shutil.rmtree(base_ner_output)

    vocab = os.path.join(model_base, model, 'vocab.txt')
    weight = os.path.join(model_base, model)
    dataset_size = dataset_info[dataset][ratio][0]
    epoch = dataset_info[dataset][ratio][1]
    train_path = os.path.join(data_base, dataset, ratio, 'train.txt')
    dev_path = os.path.join(data_base, dataset, ratio, 'dev.txt')
    test_path = os.path.join(data_base, dataset, ratio, 'test.txt')
    output_dir = f'{HOME}/output/lm/{dataset}/{ratio}'

    script_path = open(os.path.join(script_base, 'train.sh')).read()
    script = re.sub('DATASET=\n', 'DATASET=\'%s\'\n' % dataset, script_path)
    script = re.sub('ratio=\n', 'ratio=%s\n' % ratio, script)
    script = re.sub('export CUDA_DEVICE=\n', 'export CUDA_DEVICE=%s\n' % device, script)
    script = re.sub('SEED=\n', 'SEED=%s\n' % seed, script)
    script = re.sub('TASK=\n', 'TASK=\'ner\'\n', script)
    script = re.sub('dataset_size=\n', 'dataset_size=%s\n' % dataset_size, script)
    script = re.sub('export NUM_EPOCHS=\n', 'export NUM_EPOCHS=%s\n' % epoch, script)
    script = re.sub('export BERT_VOCAB=\n', 'export BERT_VOCAB=%s\n' % vocab, script)
    script = re.sub('export BERT_WEIGHTS=\n', 'export BERT_WEIGHTS=%s\n' % weight, script)
    script = re.sub('output_dir=\n', 'output_dir=%s\n' % output_dir, script)
    script = re.sub('export TRAIN_PATH=\n', 'export TRAIN_PATH=%s\n' % train_path, script)
    script = re.sub('export DEV_PATH=\n', 'export DEV_PATH=%s\n' % dev_path, script)
    script = re.sub('export TEST_PATH=\n', 'export TEST_PATH=%s\n' % test_path, script)
    script = re.sub('export GRAD_ACCUM_BATCH_SIZE=32\n', 'export GRAD_ACCUM_BATCH_SIZE=%s\n' % train_batch_size,
                    script)
    ner_script = os.path.join(script_base, 'train_%s_%s_%s.sh' % (model, dataset, ratio))
    with open(ner_script, 'w') as f:
        f.write(script)
    os.system('sh %s' % ner_script)
    os.remove(ner_script)


if __name__ == "__main__":
    models = ['biological_cased', 'computer_cased', 'bert_base_cased', 'scibert_scivocab_cased', 'biobert_cased']
    datasets = ['scierc', 'bc5cdr', 'NCBI-disease']
    seeds = [13270, 10210, 15370, 15570, 15680, 15780, 15210, 16210, 16310, 16410, 18210, 18310]
    seed = seeds[0]
    model = models[2]
    dataset = datasets[0]
    train_batch_size = 32
    run_lm(model, dataset, "0.1", seed, 3)
    # run_lm(model, dataset, "0.3", seed, 3)
    # run_lm(model, dataset, "0.5", seed, 5)
