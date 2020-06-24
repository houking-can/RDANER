import gzip
import json
import os
import random
import re
import shutil
import uuid
from multiprocessing import Pool
import time
from tqdm import tqdm
import torch


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dir_path, _, file_names in os.walk(path):
            for f in file_names:
                yield os.path.join(dir_path, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def gen_test():
    path = '/home/yhj/paper/ijcai-2020/daner/data/ner/sciie/test.txt'
    text = open(path).read()
    text = re.sub('-DOCSTART-.*?\n\n', '', text)
    text = text.split('\n\n')
    f = open('/home/yhj/paper/ijcai-2020/daner/model/test.txt', 'w')
    for sentence in text:
        if len(sentence) < 10: continue
        lines = sentence.split('\n')
        for line in lines:
            line = line.split()
            f.write('%s\t%s\n' % (line[0].ljust(25, ' '), line[-1]))
        f.write('\n')
    f.close()


def get_test():
    path = '/home/yhj/paper/ijcai-2020/daner/model/predict_results.txt'
    text = open(path).read()
    text = re.sub('\"logits\": \[\[.*\]\], \"mask\": .*?\], ', '', text)
    f = open('/home/yhj/paper/ijcai-2020/daner/model/predict.txt', 'w')
    for line in text.split('\n'):
        if len(line) < 10: continue
        line = json.loads(line)
        words = line['words']
        tags = line['tags']
        for w, t in zip(words, tags):
            f.write('%s\t%s\n' % (w, t))
        f.write('\n')
    f.close()


def compare():
    predict = '/home/yhj/paper/ijcai-2020/daner/model/predict.txt'
    test = '/home/yhj/paper/ijcai-2020/daner/model/test.txt'
    predict = open(predict).read().split('\n\n')
    test = open(test).read().split('\n\n')
    f = open('/home/yhj/paper/ijcai-2020/daner/model/compare.txt', 'w')
    for p_sent, t_sent in zip(predict, test):
        p_sent = p_sent.split('\n')
        t_sent = t_sent.split('\n')
        if len(p_sent) == len(t_sent):
            for i in range(len(t_sent)):
                try:
                    p = p_sent[i].split()[-1]
                    f.write(t_sent[i] + '\t' + p + '\n')
                except:
                    pass
            f.write('\n')
        else:
            print('error!')
    f.close()


def clean_dict():
    dictionary = json.load(open('/home/yhj/paper/ijcai-2020/daner/data/dictionary.json'))
    for split in ['train', 'test', 'dev']:
        for key, value in dictionary[split].items():
            tmp = []
            for each in value:
                tmp.append(' '.join(each.split()))
            dictionary[split][key] = tmp
    json.dump(dictionary, open('dictionary.json', 'w'), indent=4)


def pubmed_clean():
    path = '/home/yhj/paper/ijcai-2020/daner/data/corpus/pubmed-release'
    files = os.listdir(path)
    writer = open(os.path.join(path, 'biological.txt'), 'w')
    for file in files:
        print(file)
        f = open(os.path.join(path, file))
        while True:
            try:
                line = f.readline()
                line = line.replace('\n', '')
                line = json.loads(line)
                section_names = line['section_names']
                index = 0
                for i in range(len(section_names)):
                    if 'introduction' in section_names[i].lower():
                        index = i
                        break
                abstract = []
                for sent in line['abstract_text']:
                    sent = re.sub('\s+', ' ', sent)
                    sent = re.sub('<.*?>', '', sent)
                    sent = sent.strip()
                    sent = re.sub('\[[\d|,|\s]*\]', '@cite', sent)
                    abstract.append(sent)
                article = []
                for sent in line['sections'][index]:
                    sent = re.sub('\s+', ' ', sent)
                    sent = sent.strip()
                    sent = re.sub('\[[\d|,|\s]*\]', '@cite', sent)
                    article.append(sent)
                writer.write(json.dumps({'abstract': abstract, 'article': article, 'id': line['article_id']}) + '\n')

            except Exception as e:
                f.close()
                break
    writer.close()


def see_extra():
    dictionary = json.load(open('/home/yhj/paper/ijcai-2020/daner/data/dictionary.json'))
    train = dictionary['train']
    test = dictionary['test']
    dev = dictionary['dev']
    scierc_train = []
    scierc_test = []
    scierc_dev = []
    for key, value in train.items():
        scierc_train.extend(value)
    for key, value in test.items():
        scierc_test.extend(value)
    for key, value in dev.items():
        scierc_dev.extend(value)
    extra = list(set(scierc_test) - set(scierc_train))
    extra.sort()
    print(len(extra))


def gen_CDR(path, save_path, reset=False):
    print('BC5CDR:')
    split_chars = ['(', ')', '[', ']', '-', '\'', '\"', '/', '\\', ',', ':', ';']
    for file in os.listdir(path):
        writer = open(os.path.join(save_path, file), 'w')
        lines = open(os.path.join(path, file)).read().split('\n')
        bio_text = []
        tags = []
        for line in lines:
            if line == '':
                words = re.findall('[a-zA-Z]\.[a-zA-Z]\.', text)
                for word in words:
                    text = text.replace(word, word.replace('.', '#'))
                text = re.sub('\. ', ' \n', text)
                if text.endswith('.'):
                    text = text[:-1] + '\n'
                for word in words:
                    text = text.replace(word.replace('.', '#'), word)
                tags.sort(key=lambda k: k[0])

                if len(tags) == 0:
                    continue

                tmp = [tags[0]]
                i = 1
                while i < len(tags):
                    if tags[i][0] < tmp[-1][1]:
                        if tags[i][1] - tags[i][0] > tmp[-1][1] - tmp[-1][0]:
                            tmp.pop()
                            tmp.append(tags[i])
                    else:
                        tmp.append(tags[i])
                    i += 1

                i = 0
                j = 0
                while j < len(tmp) and i < len(text):
                    while i < tmp[j][0] and i < len(text):
                        if text[i] == ' ':
                            if not bio_text[-1].endswith('\n'):
                                bio_text.append('\tNN\tO\tO\n')
                        elif text[i] in split_chars:
                            if not bio_text[-1].endswith('\n'):
                                bio_text.append('\tNN\tO\tO\n')
                            bio_text.append('%s\tNN\tO\tO\n' % text[i])
                        elif text[i] == '\n':
                            if not bio_text[-1].endswith('\n'):
                                bio_text.append('\tNN\tO\tO\n')
                            bio_text.append('.\tNN\tO\tO\n\n')
                        else:
                            bio_text.append(text[i])
                        i += 1
                    label = 'B-' + tmp[j][3]
                    while i < tmp[j][1] and i < len(text):
                        if text[i] == ' ':
                            if not bio_text[-1].endswith('\n'):
                                bio_text.append('\tNN\tO\t%s\n' % label)
                                label = 'I-' + tmp[j][3]
                        elif text[i] in split_chars:
                            if not bio_text[-1].endswith('\n'):
                                bio_text.append('\tNN\tO\t%s\n' % label)
                                label = 'I-' + tmp[j][3]
                            bio_text.append('%s\tNN\tO\t%s\n' % (text[i], label))
                            label = 'I-' + tmp[j][3]
                        elif text[i] == '\n':
                            bio_text.append('.\tNN\tO\t%s\n' % label)
                            label = 'I-' + tmp[j][3]
                        else:
                            bio_text.append(text[i])
                        i += 1
                    bio_text.append('\tNN\tO\t%s\n' % label)
                    j += 1

                while i < len(text):
                    if text[i] == ' ':
                        if not bio_text[-1].endswith('\n'):
                            bio_text.append('\tNN\tO\tO\n')
                    elif text[i] in split_chars:
                        if not bio_text[-1].endswith('\n'):
                            bio_text.append('\tNN\tO\tO\n')
                        bio_text.append('%s\tNN\tO\tO\n' % text[i])
                    elif text[i] == '\n':
                        if not bio_text[-1].endswith('\n'):
                            bio_text.append('\tNN\tO\tO\n')
                        bio_text.append('.\tNN\tO\tO\n\n')
                    else:
                        bio_text.append(text[i])
                    i += 1
                bio_text = ''.join(bio_text).split('\n')
                tmp = []
                for sent in bio_text:
                    if sent.startswith('-DOCSTART-'):
                        tmp.append(sent)
                        continue
                    if sent == '':
                        tmp.append(sent)
                        continue
                    xx = sent.split('\t')
                    if len(xx) != 4:
                        continue
                    elif xx[0] == '':
                        continue
                    tmp.append(sent)

                bio_text = '\n'.join(tmp)
                if reset:
                    bio_text = re.sub('\tB-.*?\n', '\tB-Disease\n', bio_text)
                    bio_text = re.sub('\tI-.*?\n', '\tI-Disease\n', bio_text)
                writer.write(bio_text)
                bio_text = []
                tags = []
            elif re.findall('\d+\|t\|', line):
                line = line.split('|')
                bio_text.append('\n-DOCSTART- (%s)\n\n' % (line[0]))
                text = line[2] + ' '
            elif re.findall('\d+\|a\|', line):
                line = line.split('|')
                text += line[2]
            else:
                line = line.split('\t')
                if len(line) > 4:
                    tags.append((int(line[1]), int(line[2]), line[3], line[4]))

        writer.close()
        cnt = open(os.path.join(save_path, file)).read()
        cnt = re.sub('-DOCSTART-.*?\n', '\n', cnt)
        cnt = re.sub('\n\n+', '\n\n', cnt)
        print(file, cnt.count('\n\n'))


def gen_scierc():
    print('SCIERC:')
    path = '/home/yhj/paper/ijcai-2020/daner/data/tmp/scierc_json'
    save_path = '/home/yhj/paper/ijcai-2020/daner/data/ner/sciie'
    split_chars = ['(', ')', '[', ']', '-', '\'', '\"', '/', '\\', ',', ':', ';']

    for file in os.listdir(path):
        writer = open(os.path.join(save_path, file[:-4] + 'txt'), 'w')
        lines = open(os.path.join(path, file)).read().split('\n')

        for line in lines:
            if len(line) < 5: continue
            line = json.loads(line)
            bio_text = []
            bio_text.append('-DOCSTART- (%s)\n' % (line['doc_key']))

            text = []
            index = [0]
            for each in line['sentences']:
                index.append(len(each) + index[-1])
                text.extend(each)
            tags = []
            for each in line['ner']:
                tags.extend(each)
            try:
                tmp = [tags[0]]
            except:
                print(line['doc_key'])
                continue
            i = 1
            while i < len(tags):
                if tags[i][0] <= tmp[-1][1]:
                    if tags[i][1] - tags[i][0] > tmp[-1][1] - tmp[-1][0]:
                        tmp.pop()
                        tmp.append(tags[i])
                else:
                    tmp.append(tags[i])
                i += 1

            i = 0
            j = 0
            while j < len(tmp) and i < len(text):
                if i in index:
                    bio_text.append('\n\n')
                while i < tmp[j][0] and i < len(text):
                    if text[i].startswith('-') and text[i].endswith('-'):
                        chars = [text[i]]
                    else:
                        chars = re.split("(\\" + "|\\".join(split_chars) + ')', text[i])
                        if chars[0] == '': chars = chars[1:]
                        if len(chars) > 0 and chars[-1] == '': chars[:-1]
                    for char in chars:
                        bio_text.append('%s\tNN\tO\tO\n' % char)
                    i += 1
                    if i in index:
                        bio_text.append('\n\n')

                label = 'B-' + tmp[j][2]
                while i < tmp[j][1] + 1 and i < len(text):
                    if text[i].startswith('-') and text[i].endswith('-'):
                        chars = [text[i]]
                    else:
                        chars = re.split("(\\" + "|\\".join(split_chars) + ')', text[i])
                        if chars[0] == '': chars = chars[1:]
                        if len(chars) > 0 and chars[-1] == '': chars = chars[:-1]
                    for char in chars:
                        bio_text.append('%s\tNN\tO\t%s\n' % (char, label))
                        label = 'I-' + tmp[j][2]
                    i += 1
                    if i in index:
                        bio_text.append('\n\n')
                j += 1

            while i < len(text):
                if text[i].startswith('-') and text[i].endswith('-'):
                    chars = [text[i]]
                else:
                    chars = re.split("(\\" + "|\\".join(split_chars) + ')', text[i])
                    if chars[0] == '': chars = chars[1:]
                    if len(chars) > 0 and chars[-1] == '': chars = chars[:-1]
                for char in chars:
                    bio_text.append('%s\tNN\tO\tO\n' % char)
                i += 1
                if i in index:
                    bio_text.append('\n\n')

            bio_text = ''.join(bio_text).split('\n')
            tmp = []
            for sent in bio_text:
                if sent.startswith('-DOCSTART-'):
                    tmp.append(sent)
                    continue
                if sent == '':
                    tmp.append(sent)
                    continue
                xx = sent.split('\t')
                if len(xx) != 4:
                    continue
                elif xx[0] == '':
                    continue
                tmp.append(sent)

            bio_text = '\n'.join(tmp)
            writer.write(bio_text + '\n')

        writer.close()

        cnt = open(os.path.join(save_path, file[:-4] + 'txt')).read()
        cnt = re.sub('-DOCSTART-.*?\n', '\n', cnt)
        cnt = re.sub('\n\n+', '\n\n', cnt)
        print(file, cnt.count('\n\n'))


def gen_classification():
    path = '/home/yhj/paper/ijcai-2020/daner/data/split/ner'
    save_path = '/home/yhj/paper/ijcai-2020/daner/data/split/text_classification'
    for dataset in ['bc5cdr', 'NCBI-disease', 'scierc']:
        print(dataset)
        for dir in os.listdir(os.path.join(path, dataset)):
            print(dir)
            save_dir = os.path.join(save_path, dataset, dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            read_dir = os.path.join(path, dataset, dir)
            for file in os.listdir(read_dir):
                writer = open(os.path.join(save_dir, file), 'w')
                text = open(os.path.join(read_dir, file)).read()
                text = re.sub('-DOCSTART-.*?\n', '\n', text)
                text = re.sub('\n\n+', '\n\n', text)
                sentences = text.split('\n\n')
                for each in sentences:
                    if each == '': continue
                    lines = each.split('\n')
                    sent = []
                    tags = set()
                    for line in lines:
                        if line == '': continue
                        line = line.split('\t')
                        sent.append(line[0])
                        if line[-1] != 'O':
                            tags.add(line[-1][2:])
                    writer.write(json.dumps({"text": ' '.join(sent), "label": list(tags), "metadata": ""}) + '\n')

                writer.close()


def gen_biological_sentences():
    # abstract:911956
    path = '/home/yhj/paper/ijcai-2020/daner/data/corpus/biological.txt'
    save_path = '/home/yhj/paper/ijcai-2020/daner/data/corpus/biological_sentences.txt'
    f = open(path)
    writer = open(save_path, 'w')
    cnt = 0
    while True:
        try:
            line = f.readline()
            line = json.loads(line)
            for sent in line['abstract']:
                if len(sent) < 10:
                    continue
                writer.write(json.dumps({"sentence": sent, "label": [], "metadata": ""}) + '\n')
                cnt += 1
            if cnt > 1000:
                break

        except:
            break
    writer.close()
    print(cnt)


def read_CONLL(file):
    text = open(file).read()
    res = []
    if '-DOCSTART-' in text:
        docs = re.split('-DOCSTART-.*?\n', text)
        for doc in docs:
            doc = doc.strip('\n\n')
            doc = doc.strip('\n')
            if doc == '': continue
            doc = re.sub('\n\n+', '\n\n', doc)
            sentences = doc.split('\n\n')
            res.append(sentences)

    else:
        docs = re.split('\n\n+', text)
        for doc in docs:
            doc = doc.strip('\n\n')
            doc = doc.strip('\n')
            if doc == '': continue
            res.append(doc)
        res = [res]

    return res


def change_length(max_length=512):
    dataset = 'bc5cdr'
    path = '/home/yhj/paper/ijcai-2020/daner/data/ner_oracle/%s' % dataset
    save_path = '/home/yhj/paper/ijcai-2020/daner/data/ner/%s' % dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in os.listdir(path):
        cnt = 0
        docs = read_CONLL(os.path.join(path, file))
        writer = open(os.path.join(save_path, file), 'w')
        sample = []
        for doc in docs:
            for sent in doc:
                sent = sent.split('\n')
                if len(sample) + len(sent) > max_length - 2:
                    writer.write('\n'.join(sample) + '\n\n')
                    sample = sent
                    cnt += 1
                else:
                    sample.extend(sent)

            if sample != []:
                cnt += 1
                writer.write('\n'.join(sample) + '\n\n')

        writer.close()
        print(file, cnt)


def split_dataset():
    for dataset in ['scierc', 'bc5cdr', 'NCBI-disease']:
        path = '/home/yhj/paper/ijcai-2020/daner/data/ner_oracle/%s' % dataset
        for size in [0.1, 0.2, 0.3, 0.5, 1.0]:
            save_path = '/home/yhj/paper/ijcai-2020/daner/data/split/%s/%s' % (dataset, size)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            docs = read_CONLL(os.path.join(path, 'train.txt'))
            writer = open(os.path.join(save_path, 'train.txt'), 'w')
            len_docs = int(size * len(docs))
            cnt = 0
            select = random.sample(docs, len_docs)
            for doc in select:
                writer.write('-DOCSTART-\n\n')
                cnt += len(doc)
                writer.write('\n\n'.join(doc) + '\n')
            writer.close()
            shutil.copy(os.path.join(path, 'test.txt'), save_path)
            shutil.copy(os.path.join(path, 'dev.txt'), save_path)
            print(dataset, size, len_docs, len(docs), cnt)
        print('\n')


def gen_lm_corpus(ratio=0.9):
    path = '/home/yhj/paper/ijcai-2020/daner/data/corpus/'
    for domain in ['biological']:

        f = open(os.path.join(path, domain + '.txt'))
        # writer = open(save_path, 'w')
        abstracts = []
        while True:
            try:
                line = f.readline()
                line = json.loads(line)
                abstracts.append(line['title'] + ' ' + ' '.join(line['abstract']))

            except:
                break

        random.shuffle(abstracts)
        train_len = int(len(abstracts) * ratio)
        train = abstracts[:train_len]
        test = abstracts[train_len:]
        print(domain)
        print('train', len(train))
        print('test', len(test))
        print(len(abstracts))
        with open(os.path.join(path, domain, 'train.txt'), 'w') as f:
            f.write('\n\n'.join(train))
        with open(os.path.join(path, domain, 'test.txt'), 'w') as f:
            f.write('\n\n'.join(test))


def un_gz(file_name):
    # 获取文件的名称，去掉后缀名
    f_name = file_name.replace(".gz", "")
    # 开始解压
    g_file = gzip.GzipFile(file_name)
    # 读取解压后的文件，并写入去掉后缀名的同名文件（即得到解压后的文件）
    open(f_name, "wb+").write(g_file.read())
    g_file.close()
    return f_name


def extract(file):
    path = r'E:\DATASET\Pubmed'
    save_path = r'F:\DATASET\Pubmed_Abs'
    save_name = os.path.join(save_path, file.replace('xml.gz', 'txt'))
    if os.path.exists(save_name):
        print(file)
        return 1
    writer = open(save_name, 'w', encoding='utf-8')
    gz_name = os.path.join(path, file)
    xml_name = gz_name.replace(".gz", "")
    if not os.path.exists(xml_name):
        try:
            xml_name = un_gz(gz_name)
        except:
            print(file)
            return 0
    text = open(xml_name, encoding='utf-8').read()
    papers = re.findall('<PubmedArticle>(.*?)</PubmedArticle>', text, re.S)
    for paper in papers:
        PMID = re.findall('<PMID Version=\"(.*?)\">(.*?)</PMID>', paper)[0]
        abstract = re.findall('<Abstract>(.*?)</Abstract>', paper, re.S)
        title = re.findall('<ArticleTitle>(.*?)</ArticleTitle>', paper, re.S)
        if len(title) > 0:
            title = title[0]
        else:
            title = ''
        if title.startswith('[') and title.endswith('].'):
            title = title[1:-2] + '.'
        if len(abstract) > 0:
            abstract = abstract[0]
            AbstractText = re.findall('<AbstractText.*?>(.*?)</AbstractText>', abstract, re.S)
            abstract = ' '.join(AbstractText)
            writer.write(json.dumps({"title": title, "abstract": abstract, "PMID": PMID[0] + '_' + PMID[1]},
                                    ensure_ascii=False) + '\n')
    writer.close()
    os.remove(xml_name)
    print(file)
    return 1


def gen_pubmed():
    path = r'E:\DATASET\Pubmed'
    save_path = r'F:\DATASET\Pubmed_Abs'
    files = os.listdir(path)
    files = [file for file in files if file.endswith('xml.gz')]
    pool = Pool(6)
    # for file in tqdm(files):
    res = pool.map(extract, files)
    for i in res:
        print(i)
    pool.close()
    pool.join()


def judge_english(text):
    cnt = 0
    for ch in text:
        if ord(ch) < 128:
            cnt += 1
    if cnt / len(text) > 0.9:
        return True
    return False


def select_pubmed():
    # a = open(r'C:\Users\Houking\Desktop\daner\CDR_PMID.txt').read().split()
    # b = open(r'C:\Users\Houking\Desktop\daner\pubmed_pmids.txt').read().split()
    # res = []
    # j = 0
    # for i in a:
    #     while j < len(b):
    #         if b[j] == i:
    #             res.extend(b[j - 42:j + 42])
    #             break
    #         j+=1
    # extra = random.sample(b[j:],30000)
    # res.extend(extra)
    # res = set(res)
    # res = [int(i) for i in res]
    # res = res[:200000]
    # res.sort()
    # print(len(res))
    # with open('select.txt','w') as f:
    #     for i in res:
    #         f.write('%s\n' % i)
    select = open(r'/\data\tmp\select.txt').read().split()
    select = set(select)
    path = r'F:\DATASET\Pubmed_Abs'
    res = []
    for file in tqdm(os.listdir(path)):
        file = os.path.join(path, file)
        lines = open(file, encoding='utf-8').read().split('\n')
        if lines[-1] == '': lines = lines[:-1]
        for line in lines:
            paper = json.loads(line)
            if paper["PMID"][2:] in select:
                res.append(line)
                print(paper["PMID"])
    with open(r'C:\Users\Houking\Desktop\biological.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(res))


def split_sentences():
    lines = open('/home/yhj/paper/ijcai-2020/daner/data/corpus/biological.txt').read().split('\n')
    if lines[-1] == '': lines = lines[:-1]
    res = []
    for line in lines:
        paper = json.loads(line)
        text = paper['abstract']
        words = re.findall('[a-zA-Z]\.[a-zA-Z]\.', text)
        for word in words:
            text = text.replace(word, word.replace('.', '#'))
        text = re.sub('\. ', ' \n', text)
        if text.endswith('.'):
            text = text[:-1]
        for word in words:
            text = text.replace(word.replace('.', '#'), word)
        sentences = text.split('\n')
        sentences = [each + '.' for each in sentences]
        paper['abstract'] = sentences
        res.append(json.dumps(paper))
    with open('/home/yhj/paper/ijcai-2020/daner/data/biological.txt', 'w') as f:
        f.write('\n'.join(res))


def gen_baseline():
    des_path = '/home/yhj/paper/ijcai-2020/daner/data/split/ner'
    for dataset in ['scierc', 'bc5cdr', 'NCBI-disease']:
        ref_train = open(os.path.join(des_path, dataset, '1.0', 'train.txt')).read()
        train = open(os.path.join(des_path, dataset, '0.5', 'train.txt')).read()
        ref_train = set(re.split('-DOCSTART-.*?\n\n', ref_train))
        train = set(re.split('-DOCSTART-.*?\n\n', train))
        remain_train = list(ref_train - train)

        with open(os.path.join(des_path, dataset, '0.5', 'remain_train.txt'), 'w') as f:
            for each in remain_train:
                f.write('\n-DOCSTART-\n\n')
                f.write(each)


def pubtator_tokenize(path, save_path, reset=False):
    split_chars = ['(', ')', '[', ']', '-', '\'', '\"', '/', '\\', ',', ':', ';']

    for file in os.listdir(path):
        if not file.endswith('.txt'):
            continue
        lines = open(os.path.join(path, file)).read().split('\n')
        bio_text = []
        tags = []
        relations = []
        docs = []
        for line in lines:
            if line == '':
                words = re.findall('[a-zA-Z]\.[a-zA-Z]\.', text)
                for word in words:
                    text = text.replace(word, word.replace('.', '#'))
                text = re.sub('\. ', ' \n', text)
                while True:
                    text = text.rstrip(' ')
                    text = text.strip('.')
                    if not text.endswith('.') and not text.endswith(' '):
                        text += '\n'
                        break

                for word in words:
                    text = text.replace(word.replace('.', '#'), word)
                tags.sort(key=lambda k: k[0])

                if len(tags) == 0:
                    continue

                tmp = [tags[0]]
                i = 1
                while i < len(tags):
                    if tags[i][0] < tmp[-1][1]:
                        if tags[i][1] - tags[i][0] > tmp[-1][1] - tmp[-1][0]:
                            tmp.pop()
                            tmp.append(tags[i])
                    else:
                        tmp.append(tags[i])
                    i += 1

                i = 0
                j = 0
                while j < len(tmp) and i < len(text):
                    while i < tmp[j][0] and i < len(text):
                        if text[i] == ' ':
                            if not bio_text[-1].endswith('\n'):
                                bio_text.append('\tNN\tO\tO\n')
                        elif text[i] in split_chars:
                            if not bio_text[-1].endswith('\n'):
                                bio_text.append('\tNN\tO\tO\n')
                            bio_text.append('%s\tNN\tO\tO\n' % text[i])
                        elif text[i] == '\n':
                            if not bio_text[-1].endswith('\n'):
                                bio_text.append('\tNN\tO\tO\n')
                            bio_text.append('.\tNN\tO\tO\n\n')
                        else:
                            bio_text.append(text[i])
                        i += 1
                    label = 'B-' + tmp[j][3] + '_' + tmp[j][4]
                    while i < tmp[j][1] and i < len(text):
                        if text[i] == ' ':
                            if not bio_text[-1].endswith('\n'):
                                bio_text.append('\tNN\tO\t%s\n' % label)
                                label = 'I-' + tmp[j][3] + '_' + tmp[j][4]
                        elif text[i] in split_chars:
                            if not bio_text[-1].endswith('\n'):
                                bio_text.append('\tNN\tO\t%s\n' % label)
                                label = 'I-' + tmp[j][3] + '_' + tmp[j][4]
                            bio_text.append('%s\tNN\tO\t%s\n' % (text[i], label))
                            label = 'I-' + tmp[j][3] + '_' + tmp[j][4]
                        elif text[i] == '\n':
                            bio_text.append('.\tNN\tO\t%s\n' % label)
                            label = 'I-' + tmp[j][3] + '_' + tmp[j][4]
                        else:
                            bio_text.append(text[i])
                        i += 1
                    bio_text.append('\tNN\tO\t%s\n' % label)
                    j += 1

                while i < len(text):
                    if text[i] == ' ':
                        if not bio_text[-1].endswith('\n'):
                            bio_text.append('\tNN\tO\tO\n')
                    elif text[i] in split_chars:
                        if not bio_text[-1].endswith('\n'):
                            bio_text.append('\tNN\tO\tO\n')
                        bio_text.append('%s\tNN\tO\tO\n' % text[i])
                    elif text[i] == '\n':
                        if not bio_text[-1].endswith('\n'):
                            bio_text.append('\tNN\tO\tO\n')
                        bio_text.append('.\tNN\tO\tO\n\n')
                    else:
                        bio_text.append(text[i])
                    i += 1
                bio_text = ''.join(bio_text).split('\n')
                tmp = []
                if bio_text[0] == '':
                    bio_text = bio_text[1:]
                for sent in bio_text:
                    if sent == '':
                        tmp.append([])
                        continue
                    xx = sent.split('\t')
                    if len(xx) != 4:
                        continue
                    elif xx[0] == '':
                        continue
                    sent = sent.replace('\tNN\tO\t', '\t')
                    sent = sent.split('\t')
                    if sent[-1] != 'O':
                        label, DID = sent[-1].split('_')
                        sent[-1] = label
                        sent.append(DID)
                    tmp.append(sent)
                if tmp[-1] == [] and tmp[-2] == []:
                    tmp = tmp[:-1]
                docs.append({'doc': tmp, 'relations': relations, 'doc_key': doc_key})
                bio_text = []
                tags = []
                relations = []
            elif re.findall('\d+\|t\|', line):
                line = line.split('|')
                bio_text.append('\n')
                text = line[2] + ' '
            elif re.findall('\d+\|a\|', line):
                line = line.split('|')
                text += line[2]
            else:
                line = line.split('\t')
                doc_key = str(uuid.uuid1())
                if len(line) > 4:
                    label = 'Disease' if reset else line[4]
                    tags.append((int(line[1]), int(line[2]), line[3], label, line[5]))
                else:
                    relations.append((line[2], line[3]))

        json.dump(docs, open(os.path.join(save_path, file.replace('.txt', '.json')), 'w'))


def pubtator_scierc(path, save_path):
    for file in os.listdir(path):
        if file.endswith('.txt'):
            continue
        samples = []
        docs = json.load(open(os.path.join(path, file)))
        for annotation in docs:
            doc = annotation['doc']
            label_relations = annotation['relations']
            doc_key = annotation['doc_key']
            doc_tokens = []
            tokens = []
            doc_entities = []
            entities = []

            i = 0
            offset = 0
            while i < len(doc):
                token = doc[i]
                if token == []:
                    doc_tokens.append(tokens)
                    tokens = []
                    doc_entities.append(entities)
                    entities = []
                    i += 1
                    continue
                if token[-1] != 'O':
                    j = i
                    start = offset
                    type = token[1][2:]
                    DID = token[2]
                    while j < len(doc) and doc[j][-1] != 'O' and doc[j][-1] == DID:
                        tokens.append(doc[j][0])
                        offset += 1
                        j += 1
                    end = offset - 1
                    entities.append([start, end, type, DID])
                    i = j
                    continue

                tokens.append(token[0])
                offset += 1
                i += 1
            doc_relations = []
            tmp_entities = []
            for entities in doc_entities:
                relations = []
                if len(entities) > 1:
                    for i in range(len(entities) - 1):
                        for j in range(i + 1, len(entities)):
                            for relation in label_relations:
                                if entities[i][-1] == relation[0] and entities[j][-1] == relation[1]:
                                    relations.append(
                                        [entities[i][0], entities[i][1], entities[j][0], entities[j][1], "CID"])
                                    break
                                elif entities[i][-1] == relation[1] and entities[j][-1] == relation[0]:
                                    relations.append(
                                        [entities[j][0], entities[j][1], entities[i][0], entities[i][1], "CID"])
                                    break
                doc_relations.append(relations)
                tmp_entities.append([each[:-1] for each in entities])

            samples.append(
                json.dumps({"clusters": [], "sentences": doc_tokens, "ner": tmp_entities, "relations": doc_relations,
                            "doc_key": doc_key}))

        with open(os.path.join(save_path, file), 'w') as f:
            f.write('\n'.join(samples))


def pubtator_spert(path, save_path):
    for file in os.listdir(path):
        if file.endswith('.txt'):
            continue
        samples = []
        docs = json.load(open(os.path.join(path, file)))
        for annotation in docs:
            doc = annotation['doc']
            relations = annotation['relations']
            doc_key = annotation['doc_key']
            tokens = []
            entities = []

            i = 0
            offset = 0
            while i < len(doc):
                token = doc[i]
                if token == []:
                    sent_relations = []
                    if len(entities) > 1:
                        for m in range(len(entities) - 1):
                            for n in range(m + 1, len(entities)):
                                for relation in relations:
                                    if entities[m]['id'] == relation[0] and entities[n]['id'] == relation[1]:
                                        sent_relations.append({"type": "CID", "head": m, "tail": n})
                                        break
                                    elif entities[n]['id'] == relation[0] and entities[m]['id'] == relation[1]:
                                        sent_relations.append({"type": "CID", "head": n, "tail": m})
                                        break
                    samples.append({
                        "tokens": tokens,
                        "entities": entities,
                        "relations": sent_relations,
                        "ori_id": str(uuid.uuid1())})

                    tokens = []
                    entities = []
                    i += 1
                    offset = 0
                    continue
                if token[-1] != 'O':
                    j = i
                    start = offset
                    type = token[1][2:]
                    DID = token[2]
                    while j < len(doc) and doc[j][-1] != 'O' and doc[j][-1] == DID:
                        tokens.append(doc[j][0])
                        offset += 1
                        j += 1
                    end = offset
                    entities.append({"type": type, "start": start, "end": end, "id": DID})
                    i = j
                    continue

                tokens.append(token[0])
                offset += 1
                i += 1

        json.dump(samples, open(os.path.join(save_path, file), 'w'))


def strip_list(docs, chars=['', '\n', []]):
    start = 0
    for i in range(len(docs)):
        if docs[i] not in chars:
            break
        start += 1
    end = len(docs)
    for i in range(len(docs) - 1, -1, -1):
        if docs[i] not in chars:
            break
        end -= 1
    docs = docs[start:end]
    return docs


def tokenize(text):
    split_chars = ['(', ')', '[', ']', '-', '\'', '\"', '/', '\\', ',', ':', ';']
    words = re.findall('[a-zA-Z]\.[a-zA-Z]\.', text)
    words += ' '
    for word in words:
        text = text.replace(word, word.replace('.', '#'))
    text = re.sub('\.\s+', ' . \n ', text)
    sentences = text.split('\n')
    sentences = strip_list(sentences)
    res = []
    for sent in sentences:
        for word in words:
            sent = sent.replace(word.replace('.', '#'), word)
        for ch in split_chars:
            sent = sent.replace(ch, ' ' + ch + ' ')
        sent = re.split('\s+', sent)
        sent = strip_list(sent)
        res.append(sent)
    res = strip_list(res)
    return res


def select_abstracts(max_num=2000, seed=1024):
    path = '/home/yhj/paper/ijcai-2020/daner/data/LM'
    for dir in os.listdir(path):
        print(dir)
        test = open(os.path.join(path, dir, 'test.txt')).read()
        test = re.split('\n+', test)
        train = open(os.path.join(path, dir, 'train.txt')).read()
        train = re.split('\n+', train)
        random.seed(seed)
        all_abstracts = test + train
        select = random.sample(all_abstracts, max_num)
        res_classify = []
        res_ner = []
        for abstract in select:
            bio_text = []
            sentences = tokenize(abstract)
            bio_text.append('-DOCSTART-\n\n')
            for sent in sentences:
                if len(sent) < 4: continue
                res_classify.append(json.dumps({"text": ' '.join(sent), "label": [], "metadata": ""}))
                for word in sent:
                    bio_text.append(f'{word}\tNN\tO\tO\n')
                bio_text.append('\n')
            bio_text.append('\n')
            res_ner.append(''.join(bio_text))

        random.seed(seed)
        random.shuffle(res_classify)
        random.seed(seed)
        random.shuffle(res_ner)

        save_path = '/home/yhj/paper/ijcai-2020/daner/data/extra_corpus'
        with open(os.path.join(save_path, f'{dir}_classification.txt'), 'w') as f:
            f.write('\n'.join(res_classify))
        with open(os.path.join(save_path, f'{dir}_ner.txt'), 'w') as f:
            f.write('\n'.join(res_ner))


def get_log_ratio():
    path = '/home/yhj/paper/ijcai-2020/daner/output/exp_sent'
    res = {}
    for dir in os.listdir(path):
        print(dir)
        res[dir] = {}
        # ['0.1', '0.2', '0.3', '0.5']
        for ratio in ['0.1', '0.2', '0.3', '0.5', '1.0']:
            log = open(os.path.join(path, dir, ratio, 'log.txt')).read()
            tmp = re.findall(
                'corpus ratio.*?docs: (.*?), sentences: (.*?)\n.*?precision: (.*?), recall: (.*?), f1: (.*?)\n',
                log, re.S)
            score = []
            for each in tmp:
                score.append([int(each[0]), int(each[1]), float(each[2]), float(each[3]), float(each[4])])
            res[dir][ratio] = score
    json.dump(res, open(os.path.join('/home/yhj/paper/ijcai-2020/daner/results', 'ratio_' + str(time.time()) + '.json'),
                        'w'),
              indent=4)


def get_log_confidence():
    path = '/home/yhj/paper/ijcai-2020/daner/output/exp_confidence'
    res = {}
    for dir in os.listdir(path):
        print(dir)
        res[dir] = {}
        for ratio in ['0.1', '0.2', '0.3', '0.5', '1.0']:
            try:
                log = open(os.path.join(path, dir, ratio, 'log.txt')).read()
                tmp = re.findall('precision: (.*?), recall: (.*?), f1: (.*?)\n', log, re.S)
                score = []
                for each in tmp:
                    score.append([float(each[0]), float(each[1]), float(each[2])])
            except:
                score = []
            res[dir][ratio] = score
    json.dump(res, open(
        os.path.join('/home/yhj/paper/ijcai-2020/daner/results', 'confidence_' + str(time.time()) + '.json'), 'w'),
              indent=4)


def remove_state(path):
    files = iter_files(path)
    for file in files:
        if 'model.tar.gz' in file:
            print(file)
            os.remove(file)
        elif 'model_state_epoch' in file:
            print(file)
            os.remove(file)
        elif 'training_state_epoch' in file:
            print(file)
            os.remove(file)


if __name__ == "__main__":
    # path='/home/yhj/paper/ijcai-2020/daner/output/exp_confidence'
    # remove_state(path)
    # get_log_ratio()
    get_log_confidence()

    pass
