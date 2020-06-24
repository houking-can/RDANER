import os
import re
import json
from allennlp.data.dataset_readers.dataset_utils import span_utils
from tqdm import tqdm
from collections import Counter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def gen_chinese():
    path = '/home/yhj/paper/ijcai-2020/daner/data/text_classification/chinese'
    save_path = '/home/yhj/paper/ijcai-2020/daner/data/text_classification/chinese_clean'
    for file in os.listdir(path):
        print(file)
        lines = open(os.path.join(path, file)).read().split('\n')
        writer = open(os.path.join(save_path, file.replace('data', '')), 'w')
        for line in lines:
            line = line.split('\t')
            if len(line) != 3:
                continue
            text_a = clean(line[0])
            text_b = clean(line[1])
            label = line[2]

            writer.write(
                json.dumps({"text": ' '.join(text_a) + ' , ' + ' '.join(text_b), "label": [label], "metadata": ""},
                           ensure_ascii=False) + '\n')


def clean(text):
    remove_chars = ['*', '■', '★', '/', ',', '@']
    text = re.sub('\s', '', text)
    text = re.sub("\\" + "|\\".join(remove_chars), '', text)
    text = text.replace('【', '（')
    text = text.replace('】', '）')
    text = text.replace('(', '（')
    text = text.replace(')', '）')
    text = re.sub('[a-zA-Z0-9|-|_]+', '', text)
    text = text.strip('-')
    text = text.replace('（）', '')
    text = text.rstrip('（')
    text = text.lstrip('）')

    return text


def count_none():
    path = '/home/yhj/paper/ijcai-2020/daner/data/ner/'
    dirs = os.listdir(path)
    for dir in dirs:
        file = os.path.join(path, dir, 'train.txt')
        text = open(file).read()
        text = re.sub('-DOCSTART-.*?\n', '\n', text)
        text = re.sub('\n\n+', '\n\n', text)
        sentences = text.split('\n\n')
        cnt = 0
        for sent in sentences:
            if len(sent) < 3:
                continue
            sent = sent.split('\n')
            if len(sent[-1]) < 3:
                sent = sent[:-1]
            if all(line[-1] == 'O' for line in sent):
                cnt += 1
        print(dir, cnt, len(sentences))


def span_f1_measure(predict, test):
    predict = open(predict).read().strip('\n').split('\n')
    test = open(test).read()
    test = re.sub('-DOCSTART-.*?\n', '\n', test)
    test = test.strip('\n\n')
    test = re.sub('\n\n+', '\n\n', test)
    test = test.split('\n\n')
    same_cnt = 0
    error_cnt = 0
    all_cnt = 0

    all_samples = 0
    all_correct = 0
    all_predict = 0
    all_predict_improve = 0

    for i in range(len(test)):
        predict_results = json.loads(predict[i])
        predict_tags = predict_results['tags']
        sentence = test[i]
        sentence = sentence.strip('\n').split('\n')
        test_tags = []
        for line in sentence:
            line = line.split('\t')
            test_tags.append(line[-1])
        test_tags = span_utils.to_bioul(test_tags, encoding="BIO")
        test_spans = span_utils.bioul_tags_to_spans(test_tags)
        predict_spans = span_utils.bioul_tags_to_spans(predict_tags)

        all_samples += len(test_spans)
        all_correct += len(set(predict_spans) & set(test_spans))
        all_predict += len(predict_spans)
        if not all(each == 'O' for each in test_tags):
            all_predict_improve += len(predict_spans)

        if all(each == 'O' for each in test_tags):
            all_cnt += 1
            if all(each == 'O' for each in predict_tags):
                same_cnt += 1
            else:
                error_cnt += 1
    improve_precision = all_correct / all_predict_improve
    precision = all_correct / all_predict
    recall = all_correct / all_samples
    f1 = 2 * precision * recall / (precision + recall)
    f1_improve = 2 * improve_precision * recall / (improve_precision + recall)
    print('precision: %s, %s' % (precision, improve_precision))
    print('recall: %s' % recall)
    print('f1: %s, %s\n' % (f1, f1_improve))
    print('none samples:')
    print('all_cnt: %s' % all_cnt)
    print('same_cnt: %s' % same_cnt)
    print('error_cnt: %s\n' % error_cnt)


def bad_case(predict, test):
    predict = open(predict).read().strip('\n').split('\n')
    test = open(test).read()
    test = re.sub('-DOCSTART-.*?\n', '\n', test)
    test = test.strip('\n\n')
    test = re.sub('\n\n+', '\n\n', test)
    test = test.split('\n\n')
    cnt = 0
    for i in range(len(test)):
        predict_results = json.loads(predict[i])
        predict_tags = predict_results['tags']
        sentence = test[i]
        sentence = sentence.strip('\n').split('\n')
        test_tags = []
        for line in sentence:
            line = line.split('\t')
            test_tags.append(line[-1])
        test_tags = span_utils.to_bioul(test_tags, encoding="BIO")
        test_spans = span_utils.bioul_tags_to_spans(test_tags)
        predict_spans = span_utils.bioul_tags_to_spans(predict_tags)
        if test_spans != predict_spans:
            sent = predict_results['words']
            print(' '.join(sent))
            correct = list(set(test_spans) - set(predict_spans))
            wrong = list(set(predict_spans) - set(test_spans))
            correct.sort(key=lambda k: k[1][0])
            wrong.sort(key=lambda k: k[1][0])
            for i in range(len(correct)):
                correct[i] = (correct[i][0], (correct[i][1][0], correct[i][1][1]),
                              ' '.join(sent[correct[i][1][0]:correct[i][1][1] + 1]))
            for i in range(len(wrong)):
                wrong[i] = (wrong[i][0], (wrong[i][1][0], wrong[i][1][1]),
                            ' '.join(sent[wrong[i][1][0]:wrong[i][1][1] + 1]))

            print(correct)
            print(wrong)
            print('')
            cnt += 1
    return cnt


def bad_case_chinese():
    predict_path = '/home/yhj/paper/ijcai-2020/daner/predict/chinese.txt'
    test_path = '/home/yhj/paper/ijcai-2020/daner/data/text_classification/chinese/testdata.txt'
    lines = open(predict_path).read().split('\n')
    labels = open(test_path).read().split('\n')
    if labels[-1] == '': labels = labels[:-1]
    if lines[-1] == '': lines = lines[:-1]
    assert len(lines) == len(labels)
    writer = open('bad_case.txt', 'w', encoding='utf-8')
    for i in range(len(lines)):
        text_a, text_b, label = labels[i].split('\t')
        predict = json.loads(lines[i])
        class_probs = predict['class_probs']
        max_index = class_probs.index(max(class_probs))
        if label == str(max_index):
            writer.write(text_a + ' ' + text_b + ' %s\n' % label)
    writer.close()


def plot_ratio():
    path = '/home/yhj/paper/ijcai-2020/daner/results/ratio_1.json'
    res = json.load(open(path))
    save_path = '/home/yhj/paper/ijcai-2020/daner/results'
    for key, data in res.items():
        plt.title(key)
        colors = ['forestgreen', 'red', 'blue', 'black', 'purple']
        ratios = ['0.1', '0.2', '0.3', '0.5', '1.0']
        for i, ratio in enumerate(ratios):
            sentences = [each[1] for each in data[ratio]]
            f1_scores = [each[-1] for each in data[ratio]]
            plt.plot(sentences, f1_scores, color=colors[i], label=f'{int(float(ratio) * 100)}%')
        plt.legend()  # 显示图例

        plt.xlabel('Weakly Annotated Training Sentences')
        plt.ylabel('Test F1 Scores')
        plt.show()
        fig.savefig(f'{save_path}/ratio_{key}.eps', dpi=3000, format='eps')
        plt.close()


def plot_confidence():
    path = '/home/yhj/paper/ijcai-2020/daner/results/confidence_0.json'
    res = json.load(open(path))
    save_path = '/home/yhj/paper/ijcai-2020/daner/results'
    key = 'SciERC'
    data = res[key]
    fig, ax = plt.subplots()
    # plt.title(key)
    colors = ['darkorange', 'blue', 'red', 'forestgreen', 'purple']
    ratios = ['0.1', '0.2', '0.3', '0.5', '1.0'][::-1]
    shapes = ['x-', 's-', 'o-', '*-', '.-']
    for i, ratio in enumerate(ratios):
        thresholds = [0.97, 0.87, 0.77, 0.67, 0.57, 0.47, 0.37, 0.27, 0.17][::-1]

        f1_scores = [each[-1] for each in data[ratio]]
        # f1_scores = f1_scores[1:] + [initial]
        f1_scores = f1_scores[1:]
        f1_scores = f1_scores[::-1]

        plt.plot(thresholds, f1_scores, shapes[i], color=colors[i], label=f'{int(float(ratio) * 100)}%')
    plt.xlabel('Thresholds')
    plt.ylabel('Test F1 Scores')
    # plt.legend(loc=3)  # 显示图例
    plt.legend(loc=8, bbox_to_anchor=(0.3, 0))  # 显示图例
    plt.show()
    fig.savefig(f'{save_path}/{key}.eps', dpi=6000, format='eps')
    plt.close()

    key = 'NCBI-Disease'
    data = res[key]
    fig, ax = plt.subplots()
    # plt.title(key)
    colors = ['darkorange', 'blue', 'red', 'forestgreen', 'purple']
    ratios = ['0.1', '0.2', '0.3', '0.5', '1.0'][::-1]
    shapes = ['x-', 's-', 'o-', '*-', '.-']
    for i, ratio in enumerate(ratios):
        thresholds = [0.97, 0.87, 0.77, 0.67, 0.57, 0.47, 0.37, 0.27, 0.17][::-1]

        f1_scores = [each[-1] for each in data[ratio]]
        # f1_scores = f1_scores[1:] + [initial]
        f1_scores = f1_scores[1:]
        f1_scores = f1_scores[::-1]

        plt.plot(thresholds, f1_scores, shapes[i], color=colors[i], label=f'{int(float(ratio) * 100)}%')

    plt.xlabel('Thresholds')
    plt.ylabel('Test F1 Scores')
    plt.legend(loc=7,bbox_to_anchor=(1.0, 0.4))  # 显示图例
    plt.show()
    fig.savefig(f'{save_path}/{key}.eps', dpi=6000, format='eps')
    plt.close()

    key = 'BC5CDR'
    data = res[key]
    fig, ax = plt.subplots()
    # plt.title(key)
    colors = ['darkorange', 'blue', 'red', 'forestgreen', 'purple']
    ratios = ['0.1', '0.2', '0.3', '0.5', '1.0'][::-1]
    shapes = ['x-', 's-', 'o-', '*-', '.-']
    for i, ratio in enumerate(ratios):
        thresholds = [0.97, 0.87, 0.77, 0.67, 0.57, 0.47, 0.37, 0.27, 0.17][::-1]

        f1_scores = [each[-1] for each in data[ratio]]
        # f1_scores = f1_scores[1:] + [initial]
        f1_scores = f1_scores[1:]
        f1_scores = f1_scores[::-1]

        plt.plot(thresholds, f1_scores, shapes[i], color=colors[i], label=f'{int(float(ratio) * 100)}%')

    plt.xlabel('Thresholds')
    plt.ylabel('Test F1 Scores')
    plt.legend(loc=6,bbox_to_anchor=(0, 0.28))
    plt.show()
    fig.savefig(f'{save_path}/{key}.eps', dpi=6000, format='eps')
    plt.close()


if __name__ == "__main__":
    plot_confidence()
    pass
