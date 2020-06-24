import os
import re
import shutil
import argparse
import json
import random
from utils import strip_list
import copy
import torch

dataset_info = '/home/yhj/paper/ijcai-2020/daner/data/dataset_info.json'
dataset_info = json.load(open(dataset_info))['ner']
HOME = '/home/yhj/paper/ijcai-2020/daner/'


class Log():
    def __init__(self, file, add=False):
        if add:
            self.log = open(file, 'a+')
        else:
            self.log = open(file, 'w')

    def info(self, content):
        print(content)
        self.log.write(content + '\n')
        self.log.flush()

    def close(self):
        self.log.close()


def slice_list(data, data_len):
    res = []
    start = 0
    for i in range(len(data_len)):
        res.append(data[start:start + data_len[i]])
        start += data_len[i]
    return res


def correct_tags(tag_sequence, label_encoding="BIOUL"):
    right_tags = []
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == 'U':
            right_tags.append(label)
        elif label[0] == 'B':
            start = index
            flag = True
            while label[0] != 'L':
                index += 1
                if index >= len(tag_sequence):
                    flag = False
                    break
                label = tag_sequence[index]
                if not (label[0] == 'I' or label[0] == 'L'):
                    flag = False
                    break
            if flag:
                label_end = tag_sequence[start][2:]
                if all(label_end == tag_sequence[i][2:] for i in range(start, index + 1)):
                    for i in range(start, index + 1):
                        right_tags.append(tag_sequence[i])
                else:
                    for i in range(start, index + 1):
                        right_tags.append('O')

            if not flag:
                for i in range(start, index):
                    right_tags.append('O')
                index -= 1
        else:
            right_tags.append('O')
        index += 1

    return right_tags


def gen_weakly_label_dataset(args, input_file, conference):
    lines = open(args.label_output).read().split('\n')
    lines = strip_list(lines)
    _, _, predict_docs = read_conll(input_file, 1.0)

    docs_len = []
    for each in predict_docs:
        each = re.split('\n\n+', each)
        each = strip_list(each)
        docs_len.append(len(each))
    assert len(lines) == sum(docs_len)

    predict = []

    labels = open(os.path.join(args.initial_output, 'vocabulary', 'labels.txt')).read().split()

    for line in lines:
        line = json.loads(line)
        sent = []
        logits = line['logits']
        mask = line['mask']
        length = sum(mask)
        assert len(line['words']) == length
        logits = logits[:length]
        logits = torch.tensor(logits)
        logits = torch.softmax(logits, 1)
        logits = logits.tolist()
        tags = []
        for logit in logits:
            prob = max(logit)
            if prob >= conference:
                tags.append(labels[logit.index(prob)])
            else:
                tags.append('O')

        tags = correct_tags(tags)
        # for w, t in zip(line['words'], line['tags']):
        for w, t in zip(line['words'], tags):
            t = t.replace('U-', 'B-')
            t = t.replace('L-', 'I-')
            sent.append(f'{w}\tNN\tO\t{t}\n')
        predict.append(''.join(sent) + '\n')
    predict = slice_list(predict, docs_len)
    predict = ['\n'.join(each) + '\n' for each in predict]

    train = open(os.path.join(args.label_input, 'train.txt')).read()
    train = re.split('-DOCSTART-\n\n', train)
    train = strip_list(train)

    weakly_train = predict + train
    random.seed(args.seed)
    random.shuffle(weakly_train)
    with open(os.path.join(args.train_input, 'train.txt'), 'w') as f:
        for doc in weakly_train:
            f.write('\n-DOCSTART-\n\n')
            f.write(doc)
    shutil.copy(os.path.join(args.label_input, 'test.txt'), args.train_input)
    shutil.copy(os.path.join(args.label_input, 'dev.txt'), args.train_input)


def read_conll(path, ratio):
    if not os.path.exists(path):
        return 0, 0, []
    extra_corpus = open(path).read()
    docs = re.split('-DOCSTART-.*?\n+', extra_corpus)
    docs = strip_list(docs)

    docs_num = int(len(docs) * ratio) if ratio <= 1 else ratio
    docs = docs[:docs_num]
    sentences_num = 0
    for doc in docs:
        doc = re.split('\n\n+', doc)
        doc = strip_list(doc)
        sentences_num += len(doc)
    return docs_num, sentences_num, docs


def gen_combine_file(args, ratio=1.0, use_extra=False):
    input_file = os.path.join(args.label_input, f'extra_{ratio}_combine.txt')
    train = os.path.join(args.label_input, 'train.txt')
    train_docs_num, train_sentences_num, train_docs = read_conll(train, 1.0)
    if use_extra:
        extra_docs_num, extra_sentences_num, extra_docs = read_conll(args.extra_corpus, int(ratio * train_docs_num))
        combine = extra_docs
        log.info(f'Extra corpus ratio: {ratio}, docs: {extra_docs_num}, sentences: {extra_sentences_num}')
    else:
        remain_train = os.path.join(args.label_input, 'remain_train.txt')
        remain_docs_num, remain_sentences_num, remain_docs = read_conll(remain_train, ratio)
        combine = remain_docs
        log.info(f'Remain corpus ratio: {ratio}, docs: {remain_docs_num}, sentences: {remain_sentences_num}')

    with open(input_file, 'w') as f:
        for doc in combine:
            f.write('-DOCSTART-\n\n')
            f.write(doc)
    return input_file


def weakly_label(args, input_file, confidence, first):
    ner_checkpoint = args.initial_output if first else args.train_output
    cmd = ['python -m allennlp.run predict', ner_checkpoint, input_file,
           '--output-file', args.label_output,
           '--cuda-device', str(args.device),
           '--batch-size', str(args.predict_batch_size),
           '--include-package scibert',
           '--predictor sentence-tagger',
           '--use-dataset-reader',
           '--silent'
           ]
    script = open(args.label_script_path).read()
    predict_script = os.path.join(args.script_base, f'exp_label_{args.dataset}_{task}.sh')
    with open(predict_script, 'w') as f:
        f.write(script + ' '.join(cmd))
    os.system(f'sh {predict_script}')
    gen_weakly_label_dataset(args, input_file, confidence)
    os.remove(predict_script)
    return args


def train(args):
    if os.path.exists(args.train_output):
        shutil.rmtree(args.train_output)

    dataset_size = dataset_info[dataset]["1.0"][0]
    epoch = dataset_info[dataset]['1.0'][1]
    log.info(f"{dataset}-{task}, epoch:{epoch}")
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
    script = re.sub('export GRAD_ACCUM_BATCH_SIZE=32\n', 'export GRAD_ACCUM_BATCH_SIZE=%s\n' % args.train_batch_size,
                    script)

    ner_script = os.path.join(args.output, f'train_{args.dataset}_{task}.sh')
    with open(ner_script, 'w') as f:
        f.write(script)
    os.system(f'sh {ner_script}')

    os.rename(os.path.join(args.train_output, 'best.th'), os.path.join(args.train_output, 'weights.th'))
    os.remove(ner_script)
    return args


def initial_train(args):
    log.info(f"initial train...")

    if os.path.exists(args.initial_output):
        shutil.rmtree(args.initial_output)

    shutil.copy(os.path.join(args.label_input, 'train.txt'), args.train_input)
    shutil.copy(os.path.join(args.label_input, 'test.txt'), args.train_input)
    shutil.copy(os.path.join(args.label_input, 'dev.txt'), args.train_input)

    dataset_size = dataset_info[dataset][task][0]
    epoch = dataset_info[dataset][task][1]
    log.info(f"{dataset}-{task}, epoch:{epoch}")
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
    script = re.sub('output_dir=\n', 'output_dir=%s\n' % args.initial_output, script)
    script = re.sub('export TRAIN_PATH=\n', 'export TRAIN_PATH=%s\n' % train_path, script)
    script = re.sub('export DEV_PATH=\n', 'export DEV_PATH=%s\n' % dev_path, script)
    script = re.sub('export TEST_PATH=\n', 'export TEST_PATH=%s\n' % test_path, script)
    script = re.sub('export GRAD_ACCUM_BATCH_SIZE=32\n', 'export GRAD_ACCUM_BATCH_SIZE=%s\n' % args.train_batch_size,
                    script)

    initial_script = os.path.join(args.output, f'initial_train_{args.dataset}_{task}.sh')
    with open(initial_script, 'w') as f:
        f.write(script)
    os.system(f'sh {initial_script}')

    os.rename(os.path.join(args.initial_output, 'best.th'), os.path.join(args.initial_output, 'weights.th'))
    os.remove(initial_script)


if __name__ == "__main__":
    bert_models = ['biological_cased', 'computer_cased', 'bert_base_cased', 'scibert_scivocab_cased', 'biobert_cased']
    datasets = ['scierc', 'bc5cdr', 'NCBI-disease']

    iterators = 1
    dataset = datasets[2]
    task = '0.1'
    device = 3
    train_batch_size = 32
    thresholds = [0.97, 0.87, 0.77, 0.67, 0.57, 0.47, 0.37, 0.27, 0.17]
    thresholds= thresholds[6:]
    bert_model = bert_models[1] if dataset == 'scierc' else bert_models[0]
    domain = 'computer' if dataset == 'scierc' else 'biological'
    seeds = [13270, 10210, 15370, 15570, 15680, 15780, 15210, 16210, 16310, 16410, 18210, 18310]
    seed = seeds[0]
    output_base = f'{HOME}/output/exp_confidence/{dataset}/{task}'
    script_base = f'{HOME}/scripts/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=device)

    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--iterators', type=int, default=iterators)
    parser.add_argument('--predict_batch_size', type=int, default=256)
    parser.add_argument('--train_batch_size', type=int, default=train_batch_size)
    parser.add_argument('--bert_model', type=str, default=bert_model)
    parser.add_argument('--output', type=str, default=output_base)
    parser.add_argument('--label_script_path', type=str, default=f'{HOME}/scripts/predict.sh')
    parser.add_argument('--train_script_path', type=str, default=f'{HOME}/scripts/train.sh')
    parser.add_argument('--script_base', type=str, default=script_base)
    parser.add_argument('--bert_weight', type=str, default=f'{HOME}/checkpoint/{bert_model}/')
    parser.add_argument('--bert_vocab', type=str, default=f'{HOME}/checkpoint/{bert_model}/vocab.txt')
    parser.add_argument('--label_input', type=str, default=f'{HOME}/data/split/ner/{dataset}/{task}/')
    parser.add_argument('--label_output', type=str, default=os.path.join(output_base, 'predict.txt'))
    parser.add_argument('--train_input', type=str, default=os.path.join(output_base, 'train_input'))
    parser.add_argument('--initial_output', type=str, default=os.path.join(output_base, 'initial_output'))
    parser.add_argument('--train_output', type=str, default=os.path.join(output_base, 'train_output'))
    parser.add_argument('--metrics', type=str, default=os.path.join(output_base, 'metrics'))
    parser.add_argument('--extra_corpus', type=str, default=f'{HOME}/data/extra_corpus/{domain}_ner.txt')
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
    oracle_args = copy.deepcopy(args)
    log = Log(os.path.join(args.output, f'log.txt'), add=True)
    log.info(f'model:{bert_model}\tdataset:{dataset}\t{task}')
    use_extra = True if task == '1.0' else False
    input_file = gen_combine_file(args, use_extra=use_extra)

    # initial_train(args)
    metric = os.path.join(args.initial_output, 'metrics.json')
    shutil.copy(metric, os.path.join(args.metrics, f"initial_metrics.json"))
    metric = json.load(open(metric))
    log.info(f'initial:\n'
             f'precision: {metric["test_precision-overall"]}, '
             f'recall: {metric["test_recall-overall"]}, '
             f'f1: {metric["test_f1-measure-overall"]}')

    for threshold in thresholds:
        log.info(f"threshold: {threshold}")
        for i in range(args.iterators):
            first = True if i == 0 else False
            args = weakly_label(args, input_file, threshold, first=first)
            args = train(args)
            metric = os.path.join(args.train_output, 'metrics.json')
            shutil.copy(metric, os.path.join(args.metrics, f"{i}_metrics.json"))
            metric = json.load(open(metric))
            log.info(f'\titerator {i}:\n'
                     f'\tprecision: {metric["test_precision-overall"]}, '
                     f'recall: {metric["test_recall-overall"]}, '
                     f'f1: {metric["test_f1-measure-overall"]}')

    log.close()
    pass
