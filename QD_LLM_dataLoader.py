import torch
import random
from tqdm import tqdm
import mindspore.dataset as ds
# from transformers import BertModel, BertTokenizer
random.seed(3407)
PAD, CLS = '[PAD]', '[CLS]'


def build_dataset(args):
    def load_dataset(paths, pad_size=args.pad_len):
        contents = []
        for path in paths:
            with open(path, 'r', errors='ignore') as f:
                for line in tqdm(f):
                    lin = line.strip()

                    if args.n_class == 2:
                        if 'neg' in path:
                            label = 0
                        else:
                            label = 1

                    elif args.n_class == 3:
                        if 'offensive' in path:
                            label = 0
                        elif 'hate' in path:
                            label = 1
                        else:
                            label = 2

                    elif args.n_class == 4:
                        if 'business' in path:
                            label = 0
                        elif 'sci' in path:
                            label = 1
                        elif 'sport' in path:
                            label = 2
                        else:
                            label = 3

                    token = args.bert_tokenizer.tokenize(lin)
                    token = [CLS] + token
                    seq_len = len(token)
                    mask = []
                    token_ids = args.bert_tokenizer.convert_tokens_to_ids(token)

                    if pad_size:
                        if len(token) < pad_size:
                            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                            token_ids += ([0] * (pad_size - len(token)))
                        else:
                            mask = [1] * pad_size
                            token_ids = token_ids[:pad_size]
                            seq_len = pad_size

                    # teacher
                    teacher_token = args.teacher_tokenizer.tokenize(lin)
                    teacher_token = [CLS] + teacher_token
                    teacher_mask = []
                    teacher_token_ids = args.teacher_tokenizer.convert_tokens_to_ids(teacher_token)

                    if pad_size:
                        if len(teacher_token) < pad_size:
                            teacher_mask = [1] * len(teacher_token_ids) + [0] * (pad_size - len(teacher_token))
                            teacher_token_ids += ([0] * (pad_size - len(teacher_token)))
                        else:
                            teacher_mask = [1] * pad_size
                            teacher_token_ids = teacher_token_ids[:pad_size]
                            seq_len = pad_size
                    contents.append((token_ids, label, seq_len, mask, teacher_token_ids, teacher_mask, lin))
        random.shuffle(contents)
        return contents

    valid_num = -500

    if args.n_class == 2:
        training_dataset = load_dataset([args.train_cover_dir, args.train_stego_dir])
        train_data = training_dataset[:valid_num]
        valid_data = training_dataset[valid_num:]
        test_data = load_dataset([args.test_cover_dir, args.test_stego_dir])
    elif args.n_class == 3:
        training_dataset = load_dataset([args.train_offensive_dir, args.train_hate_dir, args.train_none_dir])
        train_data = training_dataset[:valid_num]
        valid_data = training_dataset[valid_num:]
        test_data = load_dataset([args.test_offensive_dir, args.test_hate_dir, args.test_none_dir])
    elif args.n_class == 4:
        training_dataset = load_dataset([args.train_business_dir, args.train_sci_dir, args.train_sport_dir, args.train_world_dir])
        train_data = training_dataset[:valid_num]
        valid_data = training_dataset[valid_num:]
        test_data = load_dataset([args.test_business_dir, args.test_sci_dir, args.test_sport_dir, args.test_world_dir])

    return train_data, valid_data, test_data


class DatasetIterater(object):
    def __init__(self, batches, args):
        self.batch_size = args.batch_size
        self.batches = batches
        self.n_batches = len(batches) // self.batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = args.device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        # return (x, seq_len, mask), y

        teacher_token_ids = torch.LongTensor([_[4] for _ in datas]).to(self.device)  
        teacher_mask = torch.LongTensor([_[5] for _ in datas]).to(self.device)  

        text = [_[6] for _ in datas]
        return (x, seq_len, mask), (teacher_token_ids, seq_len, teacher_mask), y, text

    def __next__(self):
        if self.index >= self.n_batches:  # 
            self.index = 0  # 
            raise StopIteration

       
        batches = self.batches[self.index * self.batch_size:(self.index + 1) * self.batch_size]
        self.index += 1
        batches = self._to_tensor(batches)
        return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, args):
    iters = DatasetIterater(dataset, args)
    return iters


class DatasetIterater1(object):
    def __init__(self, batches, args):
        self.batch_size = args.batch_size
        self.batches = batches
        self.n_batches = len(batches) // self.batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = args.device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        # return (x, seq_len, mask), y

        teacher_token_ids = torch.LongTensor([_[4] for _ in datas]).to(self.device)  # 
        teacher_mask = torch.LongTensor([_[5] for _ in datas]).to(self.device)  # 

        text = [_[6] for _ in datas]
        return (x, seq_len, mask), (teacher_token_ids, seq_len, teacher_mask), y, text

    def __next__(self):
        if self.index >= self.n_batches:  
            self.index = 0  
            raise StopIteration

       
        batches = self.batches[self.index * self.batch_size:(self.index + 1) * self.batch_size]
        self.index += 1
        batches = self._to_tensor(batches)
        return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator1(dataset, args):
    iters = DatasetIterater1(dataset, args)
    return iters


class TestsetIterater(object):
    def __init__(self, batches, args):
        self.batch_size = len(batches)
        self.batches = batches
        self.n_batches = len(batches) // self.batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = args.device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    # def __next__(self):
    #     if self.residue and self.index == self.n_batches:
    #         batches = self.batches[self.index * self.batch_size:len(self.batches)]
    #         self.index += 1
    #         batches = self._to_tensor(batches)
    #         return batches
    #
    #     elif self.index > self.n_batches:
    #         self.index = 0
    #         raise StopIteration
    #
    #     else:
    #         batches = self.batches[self.index * self.batch_size:(self.index + 1) * self.batch_size]
    #         self.index += 1
    #         batches = self._to_tensor(batches)
    #         return batches

    def __next__(self):
        if self.index >= self.n_batches:  
            self.index = 0 
            raise StopIteration

  
        batches = self.batches[self.index * self.batch_size:(self.index + 1) * self.batch_size]
        self.index += 1
        batches = self._to_tensor(batches)
        return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator_test(dataset, args):
    iters = TestsetIterater(dataset, args)
    return iters


# class TextData:
#     def __init__(self, neg_path, pos_path, mode):
#         self.docs = []
#         self.labels = []
#         self.label_map = {'neg': 0, 'pos': 1}
#         self.mode = mode
#         self.path = pos_path
#         self.source_len = None
#         self._load_data(neg_path, pos_path)
#
#     def _load_data(self, neg_path, pos_path):
#         with open(neg_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 self.docs.append(line.strip().split())  
#                 self.labels.append([self.label_map['neg']])  
#         with open(pos_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 self.docs.append(line.strip().split())  
#                 self.labels.append([self.label_map['pos']]) 
#
#         self.source_len = len(self.docs)
#     def __getitem__(self, index):
#         return (self.docs[index], self.labels[index])
#     def __len__(self):
#         return self.source_len
#
# def load_dataset(train_neg, train_pos, test_neg, test_pos):
#     train_data = TextData(train_neg, train_pos, 'train')
#     test_data = TextData(test_neg, test_pos, 'test')
#     imdb_train = ds.GeneratorDataset(train_data, column_names=["text", "label"], shuffle=True)
#     imdb_test = ds.GeneratorDataset(test_data, column_names=["text", "label"], shuffle=False)
#     imdb_train.source = train_data
#     imdb_test.source = test_data
#     return imdb_train, imdb_test


# class TextData:
#     def __init__(self, neg_path, pos_path, non_path, mode):
#         self.docs = []
#         self.labels = []
#         self.label_map = {'neg': 0, 'pos': 1, 'non': 2}
#         self.mode = mode
#         self.path = pos_path  
#         self.source_len = None
#         self._load_data(neg_path, pos_path, non_path)
#
#     def _load_data(self, neg_path, pos_path, non_path):
#         with open(neg_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 self.docs.append(line.strip().split())  
#                 self.labels.append([self.label_map['neg']])  
#         with open(pos_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 self.docs.append(line.strip().split()) 
#                 self.labels.append([self.label_map['pos']])  
#         with open(non_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 self.docs.append(line.strip().split())  
#                 self.labels.append([self.label_map['non']]) 
#
#         self.source_len = len(self.docs)
#     def __getitem__(self, index):
#         return (self.docs[index], self.labels[index])
#     def __len__(self):
#         return self.source_len
#
# def load_dataset(train_offensive, train_hate, train_none, test_offensive, test_hate, test_none):
#     train_data = TextData(train_offensive, train_hate, train_none, 'train')
#     test_data = TextData(test_offensive, test_hate, test_none, 'test')
#     imdb_train = ds.GeneratorDataset(train_data, column_names=["text", "label"], shuffle=True)
#     imdb_test = ds.GeneratorDataset(test_data, column_names=["text", "label"], shuffle=False)
#     imdb_train.source = train_data
#     imdb_test.source = test_data
#     return imdb_train, imdb_test


class TextData:
    def __init__(self, neg_path, pos_path, non_path, wo_path, mode):
        self.docs = []
        self.labels = []
        self.label_map = {'neg': 0, 'pos': 1, 'non': 2, 'wo': 3}
        self.mode = mode
        self.path = pos_path  
        self.source_len = None
        self._load_data(neg_path, pos_path, non_path, wo_path)

    def _load_data(self, neg_path, pos_path, non_path, wo_path):
        with open(neg_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.docs.append(line.strip().split())  
                self.labels.append([self.label_map['neg']]  
        with open(pos_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.docs.append(line.strip().split())  
                self.labels.append([self.label_map['pos']])  
        with open(non_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.docs.append(line.strip().split())  
                self.labels.append([self.label_map['non']])  
        with open(wo_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.docs.append(line.strip().split())  
                self.labels.append([self.label_map['wo']])  

        self.source_len = len(self.docs)
    def __getitem__(self, index):
        return (self.docs[index], self.labels[index])
    def __len__(self):
        return self.source_len

def load_dataset(train_offensive, train_hate, train_none, train_wo, test_offensive, test_hate, test_none, test_wo):
    train_data = TextData(train_offensive, train_hate, train_none, train_wo, 'train')
    test_data = TextData(test_offensive, test_hate, test_none, test_wo, 'test')
    imdb_train = ds.GeneratorDataset(train_data, column_names=["text", "label"], shuffle=True)
    imdb_test = ds.GeneratorDataset(test_data, column_names=["text", "label"], shuffle=False)
    imdb_train.source = train_data
    imdb_test.source = test_data
    return imdb_train, imdb_test
