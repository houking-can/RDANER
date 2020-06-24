import os
import re
import shutil
import argparse
import json
import random

dataset_info = '/home/yhj/paper/ijcai-2020/daner/data/dataset_info.json'
dataset_info = json.load(open(dataset_info))['ner']
HOME = '/home/yhj/paper/ijcai-2020/daner/'


class Log():
    def __init__(self, file):
        self.log = open(file, 'w')

    def info(self, content):
        print(content)
        self.log.write(content + '\n')

    def close(self):
        self.log.close()


def slice_list(data, data_len):
    res = []
    start = 0
    for i in range(len(data_len)):
        res.append(data[start:start + data_len[i]])
        start += data_len[i]
    return res


def gen_weakly_label_dataset(args):
    lines = open(args.label_output).read().split('\n')
    if lines[-1] == '': lines = lines[:-1]
    remain_train = open(os.path.join(args.label_input, 'remain_train.txt')).read()
    remain_train = re.split('\n-DOCSTART-\n\n', remain_train)
    if remain_train[0] == '': remain_train = remain_train[1:]
    train = open(os.path.join(args.label_input, 'train.txt')).read()
    train = re.split('-DOCSTART-\n\n', train)
    if train[0] == '': train = train[1:]
    docs_len = []
    for each in remain_train:
        docs_len.append(len(each.split('\n\n')))
    assert len(lines) == sum(docs_len)

    predict = []
    for line in lines:
        line = json.loads(line)
        sent = []
        for w, t in zip(line['words'], line['tags']):
            t = t.replace('U-', 'B-')
            t = t.replace('L-', 'I-')
            sent.append(f'{w}\tNN\tO\t{t}\n')
        predict.append(''.join(sent) + '\n')
    predict = slice_list(predict, docs_len)
    predict = ['\n'.join(each) + '\n' for each in predict]
    weakly_train = predict + train
    random.seed(args.seed)
    random.shuffle(weakly_train)
    with open(os.path.join(args.train_input, 'train.txt'), 'w') as f:
        for doc in weakly_train:
            f.write('\n-DOCSTART-\n\n')
            f.write(doc)
    shutil.copy(os.path.join(args.label_input, 'test.txt'), args.train_input)
    shutil.copy(os.path.join(args.label_input, 'dev.txt'), args.train_input)


def weakly_label(args):
    input_file = os.path.join(args.label_input, 'remain_train.txt')
    cmd = ['python -m allennlp.run predict', args.ner_weight, input_file,
           '--output-file', args.label_output,
           '--cuda-device', str(args.device),
           '--batch-size', str(args.batch_size),
           '--include-package scibert',
           '--predictor sentence-tagger',
           '--use-dataset-reader',
           '--silent'
           ]
    script = open(args.label_script_path).read()
    predict_script = os.path.join(args.script_base, f'baseline_label_{args.dataset}.sh')
    with open(predict_script, 'w') as f:
        f.write(script + ' '.join(cmd))
    os.system(f'sh {predict_script}')
    gen_weakly_label_dataset(args)
    os.remove(predict_script)
    return args


def train(args):
    if os.path.exists(args.train_output):
        shutil.rmtree(args.train_output)

    dataset_size = dataset_info[dataset]["1.0"][0]
    epoch = dataset_info[dataset]["1.0"][1]
    train_path = os.path.join(args.train_input, 'train.txt')
    dev_path = os.path.join(args.train_input, 'dev.txt')
    test_path = os.path.join(args.train_input, 'test.txt')

    script = open(args.train_script_path).read()
    script = re.sub('DATASET=\n', 'DATASET=\'%s\'\n' % args.dataset, script)
    script = re.sub('export CUDA_DEVICE=\n', 'export CUDA_DEVICE=%s\n' % args.device, script)
    script = re.sub('SEED=\n', 'SEED=%s\n' % args.seed, script)
    script = re.sub('TASK=\n', 'TASK=\'ner\'\n', script)
    script = re.sub('dataset_size=\n', 'dataset_size=%s\n' % dataset_size, script)
    script = re.sub('export NUM_EPOCHS=\n', 'export NUM_EPOCHS=%s\n' % epoch, script)
    script = re.sub('export BERT_VOCAB=\n', 'export BERT_VOCAB=%s\n' % args.bert_vocab, script)
    script = re.sub('export BERT_WEIGHTS=\n', 'export BERT_WEIGHTS=%s\n' % args.bert_weight, script)
    script = re.sub('output_dir=\n', 'output_dir=%s\n' % args.train_output, script)
    script = re.sub('export TRAIN_PATH=\n', 'export TRAIN_PATH=%s\n' % train_path, script)
    script = re.sub('export DEV_PATH=\n', 'export DEV_PATH=%s\n' % dev_path, script)
    script = re.sub('export TEST_PATH=\n', 'export TEST_PATH=%s\n' % test_path, script)

    ner_script = os.path.join(args.script_base, f'baseline_train_{args.dataset}.sh')
    with open(ner_script, 'w') as f:
        f.write(script)
    os.system(f'sh {ner_script}')

    os.rename(os.path.join(args.train_output, 'best.th'), os.path.join(args.train_output, 'weights.th'))
    args.ner_weight = args.train_output
    os.remove(ner_script)
    return args


if __name__ == "__main__":
    bert_models = ['biological_cased', 'computer_cased', 'bert_base_cased', 'scibert_scivocab_cased','biobert_cased']
    bert_model = bert_models[0]
    datasets = ['scierc', 'bc5cdr', 'NCBI-disease']
    dataset = datasets[1]
    seeds = [13270, 10210, 15370, 15570, 15680, 15780, 15210, 16210, 16310, 16410, 18210, 18310]
    seed = seeds[1]
    output_base = f'{HOME}/output/baselines/{dataset}'
    script_base = f'{HOME}/scripts/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--iterators', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--bert_model', type=str, default=bert_model)
    parser.add_argument('--output', type=str, default=output_base)
    parser.add_argument('--label_script_path', type=str, default=f'{HOME}/scripts/predict.sh')
    parser.add_argument('--train_script_path', type=str, default=f'{HOME}/scripts/train.sh')
    parser.add_argument('--script_base', type=str, default=script_base)
    parser.add_argument('--ner_weight', type=str, default=f'{HOME}/best/ner/{dataset}/0.1/')
    parser.add_argument('--bert_weight', type=str, default=f'{HOME}/checkpoint/{bert_model}/')
    parser.add_argument('--bert_vocab', type=str, default=f'{HOME}/checkpoint/{bert_model}/vocab.txt')
    parser.add_argument('--label_input', type=str, default=f'{HOME}/data/split/ner/{dataset}/0.1/')
    parser.add_argument('--label_output', type=str, default=os.path.join(output_base, 'predict.txt'))
    parser.add_argument('--train_input', type=str, default=os.path.join(output_base, 'train_input'))
    parser.add_argument('--train_output', type=str, default=os.path.join(output_base, 'train_output'))
    parser.add_argument('--metrics', type=str, default=os.path.join(output_base, 'metrics'))
    args = parser.parse_args()

    # Initial settings
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(args.metrics):
        os.makedirs(args.metrics)
    if not os.path.exists(args.train_output):
        os.makedirs(args.train_output)
    if not os.path.exists(args.train_input):
        os.makedirs(args.train_input)

    log = Log(os.path.join(args.output, f'log.txt'))
    log.info(f'model:{bert_model}\tdataset:{dataset}')

    for i in range(args.iterators):
        log.info(f"Iterator {i}/{args.iterators}:")
        log.info(f"weakly label...")
        args = weakly_label(args)
        log.info(f"train ner using weakly label...")
        args = train(args)
        metric = os.path.join(args.train_output, 'metrics.json')
        shutil.copy(metric, os.path.join(args.metrics, f"{i}_metrics.json"))
        metric = json.load(open(metric))
        log.info(f'dataset {dataset}, iterator {i}:\n'
                 f'precision: {metric["test_precision-overall"]}, '
                 f'recall: {metric["test_recall-overall"]}, '
                 f'f1: {metric["test_f1-measure-overall"]}')

    log.close()
