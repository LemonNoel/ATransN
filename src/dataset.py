import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
import bisect
from torch.utils.data import Dataset


def get_true_head_and_tail(triples):
    '''
    Build a dictionary of true triples that will
    be used to filter these true triples for negative sampling
    '''

    true_head = {}
    true_tail = {}

    for head, relation, tail in triples:
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    for relation, tail in true_head:
        true_head[(relation, tail)] = np.array(
            list(set(true_head[(relation, tail)])))
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(
            list(set(true_tail[(head, relation)])))

    return true_head, true_tail


class TrainDataset(Dataset):
    def __init__(self, triples, num_entities, num_relations, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = 1/subsampling_weight*0.001

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(
                self.num_entities, size=self.negative_sample_size * 2, dtype=np.int64)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError(
                    'Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.tensor([_[0] for _ in data])
        negative_sample = torch.from_numpy(np.vstack([_[1] for _ in data]))
        subsample_weight = torch.tensor([_[2] for _ in data])

        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, num_entities, num_relations, mode):
        self.len = len(triples)
        self.triples = triples
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.mode = mode

        if self.mode == "head-batch":
            self.all_true_triples = sorted([x[::-1] for x in all_true_triples]) # tail, relation, head
        elif self.mode == 'tail-batch': 
            self.all_true_triples = sorted(all_true_triples) # head, relation, tail
        else:
            raise ValueError(
                'negative batch mode %s not supported' % self.mode)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        filter_bias = np.zeros((self.num_entities), dtype=np.float32)
        if self.mode == "head-batch":
            i = bisect.bisect_left(self.all_true_triples, (tail, relation, -1))
            # j = bisect.bisect_left(self.all_true_triples, (tail, relation, self.num_entities)) # O(nlogn)
            # for k in range(i, j):
                # filter_bias[self.all_true_triples[k][2]] = -1
            for j in range(i, len(self.all_true_triples)):
                triple = self.all_true_triples[j]
                if triple[0] != tail and triple[1] != relation:
                    break
                filter_bias[triple[2]] = -1e8 # exclude positive samples
            filter_bias[head] = 0.0
        elif self.mode == 'tail-batch':
            i = bisect.bisect_left(self.all_true_triples, (head, relation, -1))
            # j = bisect.bisect_left(self.all_true_triples, (head, relation, self.num_entities), lo=i) # O(nlogn)
            # for k in range(i, j):
                # filter_bias[self.all_true_triples[k][2]] = -1
            for j in range(i, len(self.all_true_triples)):
                triple = self.all_true_triples[j]
                if triple[0] != head and triple[1] != relation:
                    break
                filter_bias[triple[2]] = -1e8 # exclude positive samples
            filter_bias[tail] = 0.0 # include current sample
        else:
            raise ValueError(
                'negative batch mode %s not supported' % self.mode)

        return np.array(self.triples[idx]), filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.from_numpy(np.vstack([x[0] for x in data]))
        filter_bias = torch.from_numpy(np.vstack([x[1] for x in data]))
        mode = data[0][2]
        return positive_sample, filter_bias, mode


class BidirectionalOneShotIterator(object):

    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


class KGDataset(object):
    def __init__(self, dict_path="", model_path="", data_path=""):
        # dict {id: string} #
        if dict_path:
            self.ent_dict = self.load_dict(os.path.join(dict_path, 'entity_dict.txt'))
            self.rel_dict = self.load_dict(os.path.join(dict_path, 'relation_dict.txt'))
        else:
            self.ent_dict = {}
            self.rel_dict = {}

        # list size=(name, embedding_dim) #
        if model_path:
            if os.path.exists(os.path.join(model_path, 'entity2vec.vec')) and \
                os.path.exists(os.path.join(model_path, 'relation2vec.vec')):
                self.entity_embedding = self.load_vec(os.path.join(model_path, 'entity2vec.vec'))
                self.relation_embedding = self.load_vec(os.path.join(model_path, 'relation2vec.vec'))
            elif os.path.exists(os.path.join(model_path, 'entity.pt')) and \
                os.path.exists(os.path.join(model_path, 'relation.pt')):
                self.entity_embedding = self.load_model(os.path.join(model_path, 'entity.pt'))
                self.relation_embedding = self.load_model(os.path.join(model_path, 'relation.pt'))
            elif os.path.exists(os.path.join(model_path, 'checkpoint_valid.pt')):
                model = self.load_model(os.path.join(model_path, 'checkpoint_valid.pt'))
                self.entity_embedding = model["entity_embedding"]
                self.relation_embedding = model["relation_embedding"]
            elif os.path.exists(os.path.join(model_path, 'checkpoint_test.pt')):
                model = self.load_model(os.path.join(model_path, 'checkpoint_test.pt'))
                self.entity_embedding = model["entity_embedding"]
                self.relation_embedding = model["relation_embedding"]
            else:
                raise FileNotFoundError
        else:
            self.entity_embedding = None
            self.relation_embedding = None

        # list [[h, r, t]] #
        if data_path:
            self.train_set = self.load_dataset(os.path.join(data_path, 'train_triple_id.txt'))
            self.valid_set = self.load_dataset(os.path.join(data_path, 'valid_triple_id.txt'))
            self.test_set = self.load_dataset(os.path.join(data_path, 'test_triple_id.txt'))
        else:
            self.train_set = []
            self.valid_set = []
            self.test_set = []

    def get_entity_count(self):
        return len(self.ent_dict) if self.ent_dict else 0

    def get_relation_count(self):
        return len(self.rel_dict) if self.rel_dict else 0

    def get_embedding_dim(self):
        # return self.entity_embedding.weight.size()[1] if self.entity_embedding is not None else -1
        return self.entity_embedding.size(1) if self.entity_embedding is not None else -1

    def get_train_size(self):
        return len(self.train_set) if self.train_set is not None else 0

    def get_valid_size(self):
        return len(self.valid_set) if self.valid_set is not None else 0

    def get_test_size(self):
        return len(self.test_set) if self.test_set is not None else 0

    @staticmethod
    def _read_dictionary(filename, reverse=False):
        def str_int(x):
            item = (int(x[1]), x[0]) if reverse else (x[0], int(x[1]))
            return item
        with open(filename, 'r', encoding='utf-8') as fp:
            dictionary = dict()
            for i, x in enumerate(fp):
                x = str_int(x.strip().split('\t'))
                dictionary[x[0]] = x[1]
        return dictionary

    def load_dict(self, filename):
        return self._read_dictionary(filename, reverse=True)

    def load_vec(self, filename):
        vecs = list()
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if i >= len(self.ent_dict):
                    break
                vec = np.array(line.strip().split("\t"), dtype=np.float32)
                vecs.append(vec)
        vecs = torch.from_numpy(np.vstack(vecs))
        return vecs

    def load_model(self, filename):
        model = torch.load(filename, map_location=torch.device("cpu"))
        if isinstance(model, dict):
            return model["learner"]
        elif isinstance(model, torch.Tensor):
            return model
        else:
            raise NotImplementedError

    def load_dataset(self, filename):
        data = []
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                x = [int(tmp) for tmp in line.strip().split('\t')]
                if x[1] not in self.rel_dict or x[0] not in self.ent_dict or x[2] not in self.ent_dict:
                    continue
                data.append(tuple(x))
        return data


def batch_iter(data, batch_size=64):
    """generate a batch iterator"""
    data_len = len(data)
    data_shuffle = data.sample(frac=1).reset_index(drop=True)

    for i in range(0, data_len, batch_size):
        j = i + batch_size
        yield data_shuffle.iloc[i:j]


def load_shared_entity(filename, separator='\t'):
    """load shared entitites"""
    # TODO DELETE TOP ONE ENTITIES
    # top_entities = pickle.load(open('../intermediate/top_one_entity_transe_200.pkl', 'rb'))
    shared_entity = pd.read_csv(
        filename, sep=separator, header=None, names=['teacher', 'student'])
    # shared_entity = shared_entity[(True ^ shared_entity['student'].isin(top_entities))]
    return shared_entity


def get_learner_iter_per_shared(shared_len, triple_len, shared_batch, triple_batch):
    """get the integer ratio of the triple size and shared entity size"""
    shared_num = math.ceil(shared_len/shared_batch)
    triple_num = math.ceil(triple_len/triple_batch)
    num_iter = math.ceil(triple_num/shared_num)
    return num_iter
