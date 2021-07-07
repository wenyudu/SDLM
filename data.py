import os
import torch
import nltk
import pickle
import random

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class PTBLoader(object):
    '''Data path is assumed to be a directory with
       pkl files and a corpora subdirectory.
    '''
    def __init__(self, data_path,use_gold_POS_tagger=True):
        # make path available for nltk
        nltk.data.path.append(data_path)
        dict_filepath = os.path.join(data_path, 'dict.pkl')
        if use_gold_POS_tagger:
            train_data_filepath = os.path.join(data_path, 'parsed.pkl')
        else:
            train_data_filepath = os.path.join(data_path, 'StanfordPOS_parsed.pkl')
        eval_data_filepath = os.path.join(data_path,'eval.pkl')

        print("loading dictionary ...")
        self.dictionary = pickle.load(open(dict_filepath, "rb"))

        # build tree and distance
        print("loading data ...")
        file_data = open(train_data_filepath, 'rb')
        self.train, self.arc_dictionary, self.stag_dictionary = pickle.load(file_data)
        file_data.close()
        file_data = open(eval_data_filepath, 'rb')
        self.valid, self.test = pickle.load(file_data)
        file_data.close()

    def batchify(self, dataname, batch_size, dratio,sort=False):
        sents, trees = None, None
        if dataname == 'train':
            idxs, tags, stags, arcs, distances, sents, trees = self.train
        elif dataname == 'valid':
            idxs, tags, stags, arcs, distances, _, _ = self.valid
        elif dataname == 'test':
            idxs, tags, stags, arcs, distances, _, _ = self.test
        else:
            raise('need a correct dataname')

        assert len(idxs) == len(distances)
        assert len(idxs) == len(tags)

        length = len(idxs)

        return idxs[:int(dratio*length)], distances[:int(dratio*length)]


def syntactic_penn(args, batch_size,dratio=1.0):
    def flat(l): return [item for sublist in l for item in sublist]

    print("loading penn data ...")
    data = PTBLoader(data_path=args.data)

    train_idx, train_distance = data.batchify('train', batch_size,dratio)
    max_dist = max(flat(train_distance))
    for c, i in enumerate(train_distance):
        i.insert(0, max_dist + 1)

    def batchify(data, bsz, random_start_idx=0):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        start_idx = random_start_idx
        data = data.narrow(0, start_idx, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        if args.cuda:
            data = data.cuda()
        return data

    random_start_idx = random.randint(0, len(flat(train_idx)) % batch_size - 1)
    train_data = batchify(torch.LongTensor(flat(train_idx)), batch_size, random_start_idx=random_start_idx)
    train_dist = batchify(torch.Tensor(flat(train_distance)), batch_size, random_start_idx=random_start_idx)
    val_data = batchify(data.valid, 80)
    test_data = batchify(data.test, 1)

    return (train_data,train_dist,val_data,test_data,data.dictionary)

