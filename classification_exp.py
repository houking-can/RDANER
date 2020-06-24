import os
import re
import shutil
import json

HOME = '/home/yhj/paper/ijcai-2020/daner'
dataset_info = f'{HOME}/data/dataset_info.json'
dataset_info = json.load(open(dataset_info))['text_classification']

model_base = f'{HOME}/checkpoint'
data_base = f'{HOME}/data/split/text_classification'
script_base = f'{HOME}/scripts'

def run_classification(model, dataset, ratio, seed, device):
    print(model, dataset, ratio, seed, device)
    output_dir = f'{HOME}/output/text_classification/{dataset}/{ratio}'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    vocab = os.path.join(model_base, model, 'vocab.txt')
    weight = os.path.join(model_base, model)
    dataset_size = dataset_info[dataset][ratio][0]
    epoch = dataset_info[dataset][ratio][1]
    train_path = os.path.join(data_base, dataset, ratio, 'train.txt')
    dev_path = os.path.join(data_base, dataset, ratio, 'dev.txt')
    test_path = os.path.join(data_base, dataset, ratio, 'test.txt')

    script = open(os.path.join(script_base,'train.sh')).read()
    script = re.sub('DATASET=\n', 'DATASET=\'%s\'\n' % dataset, script)
    script = re.sub('ratio=\n', 'ratio=%s\n' % ratio, script)
    script = re.sub('export CUDA_DEVICE=\n', 'export CUDA_DEVICE=%s\n' % device, script)
    script = re.sub('SEED=\n', 'SEED=%s\n' % seed, script)
    script = re.sub('TASK=\n', 'TASK=\'text_classification\'\n', script)
    script = re.sub('dataset_size=\n', 'dataset_size=%s\n' % dataset_size, script)
    script = re.sub('export NUM_EPOCHS=\n', 'export NUM_EPOCHS=%s\n' % epoch, script)
    script = re.sub('export BERT_VOCAB=\n', 'export BERT_VOCAB=%s\n' % vocab, script)
    script = re.sub('export BERT_WEIGHTS=\n', 'export BERT_WEIGHTS=%s\n' % weight, script)
    script = re.sub('output_dir=\n', 'output_dir=%s\n' % output_dir, script)
    script = re.sub('export TRAIN_PATH=\n', 'export TRAIN_PATH=%s\n' % train_path, script)
    script = re.sub('export DEV_PATH=\n', 'export DEV_PATH=%s\n' % dev_path, script)
    script = re.sub('export TEST_PATH=\n', 'export TEST_PATH=%s\n' % test_path, script)


    ner_script = os.path.join(script_base, 'train_%s_%s_%s.sh' % (model, dataset, ratio))
    with open(ner_script, 'w') as f:
        f.write(script)
    os.system('sh %s' % ner_script)
    os.remove(ner_script)


if __name__ == "__main__":
    models = ['computer_cased', 'biological_cased', 'bert_base_cased', 'scibert_scivocab_cased', 'biobert_cased']
    datasets = ['scierc', 'bc5cdr', 'NCBI-disease']
    seeds = [13270, 10210, 15370, 15570, 15680, 15780, 15210, 16210, 16310, 16410, 18210, 18310]
    seed = seeds[0]
    model = models[1]
    dataset = datasets[1]
    run_classification(model, dataset, "1.0", seed, 3)
    # run_classification(model, dataset, "1.0", seed, 0)
