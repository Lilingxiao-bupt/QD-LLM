import os
import time
import torch
from glob import glob
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer
import argparse

import DataLoader
import QNLP_network

# transformers: 4.29.0  # QLLM
# peft: 0.3.0
# transformers: 4.40.0  # LLM
# peft: 0.5.0

class JSDivLoss(nn.Cell):
    def __init__(self):
        super(JSDivLoss, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='mean')
        self.log_softmax = nn.LogSoftmax(axis=1)
        self.softmax = nn.Softmax(axis=1)

    def construct(self, student_output, teacher_output):
        m = 0.5 * (self.softmax(student_output) + self.softmax(teacher_output))
        kl_div1 = self.kl_div(self.log_softmax(student_output), m)
        kl_div2 = self.kl_div(self.log_softmax(teacher_output), m)
        js_div = 0.5 * (kl_div1 + kl_div2)
        return js_div


def forward_fn(data, teacher_text, label):
    # label = mnp.reshape(label, (-1,))
    # label = label.astype(mnp.int32)
    # label = label.cpu().numpy()
    label = ms.Tensor(label.cpu().numpy(), dtype=ms.int32)
    label = mnp.array(label, mnp.int32)
    # data0 = ms.Tensor(inputs[0].cpu().numpy(), dtype=ms.int32)

    student_outpu = model(data)
    teacher_output = tearcher_model(teacher_text)

    T = 2
    alpha = args.alpha
    # student_output = student_output.asnumpy()
    # student_output = torch.from_numpy(student_output)
    SM = nn.Softmax(axis=-1)
    student_output = SM(student_outpu)
    LS = nn.LogSoftmax(axis=1)
    student_output = LS(student_outpu / T)
    teacher_output = F.softmax(teacher_output / T, dim=1)
    teacher_output = teacher_output.cpu()
    teacher_output = Tensor(teacher_output.detach().numpy(), dtype=mstype.float32)

    loss1 = loss_fn1(student_outpu, label)
    loss2 = loss_fn2(student_output, teacher_output) * T * T
    loss3 = loss_fn3(student_output, teacher_output)
    # loss_total = (1-alpha) * loss2 + (1-alpha) * loss3 * 100
    loss_total = (1-alpha) * loss1 + alpha * loss2 + alpha * loss3
    # loss_total = loss1 + loss2 + loss3
    return loss_total


def train_step(data, teacher_text, label):
    loss, grads = grad_fn(data, teacher_text, label)
# def train_step(data, label):
#     loss, grads = grad_fn(data, label)
    optimizer(grads)
    return loss


# def train_one_epoch(model, train_dataset, epoch=0):
def train_one_epoch(model, train_iter, epoch=0):
    model.set_train()
    # total = train_dataset.get_dataset_size()
    total = len(train_iter)
    loss_total = 0
    step_total = 0
    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for feature, teacher_feature, target, text in train_iter:
            loss = train_step(feature, text, target)
        # for feature, target in train_iter:
        #     loss = train_step(feature, target)
            loss_total += loss.asnumpy()
            step_total += 1
            t.set_postfix(loss=loss_total/step_total)
            t.update(1)


def calculate_metrics(preds, y):
    preds = preds.asnumpy()
    y = y.asnumpy()
    precision = metrics.precision_score(y, preds, average='weighted', zero_division=0)
    recall = metrics.recall_score(y, preds, average='weighted')
    # TP = mnp.sum((preds == 1) & (y == 1)).astype(float)
    # FP = mnp.sum((preds == 1) & (y == 0)).astype(float)
    # FN = mnp.sum((preds == 0) & (y == 1)).astype(float)
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    if precision + recall == 0:
        F1 = 0.0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)
    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = round(F1, 4)
    # precision = round(precision.asnumpy().item(), 4)
    # recall = round(recall.asnumpy().item(), 4)
    # F1 = round(F1.asnumpy().item(), 4)
    # return round(precision.asnumpy(), 4), round(recall.asnumpy(), 4), round(F1.asnumpy(), 4)
    return precision, recall, F1


def binary_accuracy(preds, y):
    # rounded_preds = np.around(ops.sigmoid(preds).asnumpy())
    # correct = (rounded_preds == y).astype(np.float32)
    # acc = correct.sum() / len(correct)
    # probs = ops.sigmoid(preds)

    # output_label = []
    # for j in preds:
    #     if j[0] > j[1]:
    #         output_label.append(0)
    #     else:
    #         output_label.append(1)
    # output_label = Tensor(output_label, ms.float32)
    # correct = output_label == y
    # result = correct.astype(int)
    # SM = ops.Softmax()
    # probs = SM(preds)
    # pred_classes = ops.Argmax(axis=1)(probs)
    # y = y.astype(mnp.int32)
    # correct = pred_classes == y
    # correct = correct.astype(mnp.float32)

    # correct = result.astype(mnp.float32)
    # acc = mnp.mean(correct)
    # acc = round(acc.asnumpy().item(), 4)
    # precision, recall, F1 = calculate_metrics(output_label, y)

    probs = ops.sigmoid(preds)
    pred_classes = ops.Argmax(axis=1)(probs)
    correct = pred_classes == y.astype(mnp.int32)  
    correct = correct.astype(mnp.float32)
    acc = mnp.mean(correct)
    acc = round(acc.asnumpy().item(), 4) 


    precision, recall, F1 = calculate_metrics(pred_classes, y)
    return acc, precision, recall, F1


def evaluate(model, test_dataset, criterion, epoch=0):
    # total = test_dataset.get_dataset_size()
    total = len(test_dataset)
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_F1 = 0
    step_total = 0
    model.set_train(False)
    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        # for i in test_dataset.create_tuple_iterator():
        for feature, teacher_feature, labels, text in test_dataset:
        # for feature, labels in test_dataset:
            # feature = ms.Tensor(feature[0].cpu().numpy(), dtype=ms.int32)
            predictions = model(feature)
            # labels = i[1]
            labels = ms.Tensor(labels.cpu().numpy(), dtype=ms.int32)
            labels = mnp.array(labels, mnp.int32)
            # labels = mnp.reshape(labels, (-1,))
            # labels = labels.astype(mnp.int32)
            loss = criterion(predictions, labels)
            # print(predictions, labels)
            epoch_loss += loss.asnumpy()
            acc, precision, recall, F1 = binary_accuracy(predictions, labels)
            epoch_acc += acc
            epoch_precision += precision
            epoch_recall += recall
            epoch_F1 += F1
            step_total += 1
            t.set_postfix(loss=epoch_loss/step_total, acc=epoch_acc/step_total, precision=epoch_precision/step_total,
                          recall=epoch_recall/step_total, F1=epoch_F1/step_total)
            t.update(1)
    return epoch_loss/step_total, epoch_acc/step_total, epoch_precision/step_total, epoch_recall/step_total, epoch_F1/step_total


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

# data_name = "senti"      # Senti140
data_name = "stega"   # Steganalysis
# data_name = "hate"    # Hate speech detection
# data_name = "topic"     # Thematic analysis
# data_name = "wino"     # Thematic analysis

train_neg_path = './Dataset/' + data_name + '/train_neg.txt'
train_pos_path = './Dataset/' + data_name + '/train_pos.txt'
test_neg_path = './Dataset/' + data_name + '/test_neg.txt'
test_pos_path = './Dataset/' + data_name + '/test_pos.txt'
# imdb_trai, imdb_tes = load_dataset(train_neg_path, train_pos_path, test_neg_path, test_pos_path)

# train_offensive_path = './Dataset/' + data_name + '/train_offensive.txt'
# train_hate_path = './Dataset/' + data_name + '/train_hate.txt'
# train_none_path = './Dataset/' + data_name + '/train_none.txt'
# test_offensive_path = './Dataset/' + data_name + '/test_offensive.txt'
# test_hate_path = './Dataset/' + data_name + '/test_hate.txt'
# test_none_path = './Dataset/' + data_name + '/test_none.txt'
# imdb_trai, imdb_tes = load_dataset(train_offensive_path, train_hate_path, train_none_path,
#                                     test_offensive_path, test_hate_path, test_none_path)

# train_business_path = './Dataset/' + data_name + '/train_business.txt'
# train_sci_path = './Dataset/' + data_name + '/train_sci.txt'
# train_sport_path = './Dataset/' + data_name + '/train_sport.txt'
# train_world_path = './Dataset/' + data_name + '/train_world.txt'
# test_business_path = './Dataset/' + data_name + '/test_business.txt'
# test_sci_path = './Dataset/' + data_name + '/test_sci.txt'
# test_sport_path = './Dataset/' + data_name + '/test_sport.txt'
# test_world_path = './Dataset/' + data_name + '/test_world.txt'
# imdb_trai, imdb_tes = load_dataset(train_business_path, train_sci_path, train_sport_path, train_world_path,
#                                    test_business_path, test_sci_path, test_sport_path, test_world_path)

parser = argparse.ArgumentParser(description='QNLP')
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch.')

parser.add_argument('-train-cover-dir', type=str, default=train_neg_path, help='the path of train cover data. [default: tweets_cover.txt]')
parser.add_argument('-train-stego-dir', type=str, default=train_pos_path, help='the path of train stego data. [default: tweets_stego.txt]')
parser.add_argument('-test-cover-dir', type=str, default=test_neg_path, help='the path of test cover data. [default: test_cover.txt]')
parser.add_argument('-test-stego-dir', type=str, default=test_pos_path, help='the path of test stego data. [default: test_stego.txt]')

# parser.add_argument('-train-offensive-dir', type=str, default=train_offensive_path, help='the path of train cover data. [default: tweets_cover.txt]')
# parser.add_argument('-train-hate-dir', type=str, default=train_hate_path, help='the path of train stego data. [default: tweets_stego.txt]')
# parser.add_argument('-train-none-dir', type=str, default=train_none_path, help='the path of train stego data. [default: tweets_stego.txt]')
# parser.add_argument('-test-offensive-dir', type=str, default=test_offensive_path, help='the path of test cover data. [default: test_cover.txt]')
# parser.add_argument('-test-hate-dir', type=str, default=test_hate_path, help='the path of test stego data. [default: test_stego.txt]')
# parser.add_argument('-test-none-dir', type=str, default=test_none_path, help='the path of test stego data. [default: test_stego.txt]')

# parser.add_argument('-train-business-dir', type=str, default=train_business_path, help='the path of train cover data. [default: tweets_cover.txt]')
# parser.add_argument('-train-sci-dir', type=str, default=train_sci_path, help='the path of train stego data. [default: tweets_stego.txt]')
# parser.add_argument('-train-sport-dir', type=str, default=train_sport_path, help='the path of train stego data. [default: tweets_stego.txt]')
# parser.add_argument('-train-world-dir', type=str, default=train_world_path, help='the path of train stego data. [default: tweets_stego.txt]')
# parser.add_argument('-test-business-dir', type=str, default=test_business_path, help='the path of test cover data. [default: test_cover.txt]')
# parser.add_argument('-test-sci-dir', type=str, default=test_sci_path, help='the path of test stego data. [default: test_stego.txt]')
# parser.add_argument('-test-sport-dir', type=str, default=test_sport_path, help='the path of test stego data. [default: test_stego.txt]')
# parser.add_argument('-test-world-dir', type=str, default=test_world_path, help='the path of test stego data. [default: test_stego.txt]')

parser.add_argument('-device', type=str, default='cpu', help='device to use for training [default:cuda]')
parser.add_argument('-hidden-size', type=int, default=256)
parser.add_argument('-output-size', type=int, default=2)
parser.add_argument('-num-layers', type=int, default=2)

parser.add_argument('-num_epochs', type=int, default=0, help='number of epochs for train [default:20]')
parser.add_argument('-batch_size', type=int, default=1, help='number of epochs for train [default:20]')
parser.add_argument('-lr', type=float, default=0.06)
parser.add_argument('-pad-len', type=int, default=60)
parser.add_argument('-qbits', type=int, default=11, help='11')
parser.add_argument('-n-class', type=int, default=2, help='2,3,4')
parser.add_argument('-withlabel', type=str, default='no', help='yes,no')
parser.add_argument('-alpha', type=float, default=0.5, help='yes,no')
parser.add_argument('-LLM', type=str, default='LLaMA2-7B', help='BLOOMZ-1.1B, BLOOMZ-3B, OPT-6.7B, LLaMA2-7B, LLaMA3-8B')
args = parser.parse_args()

# bert_based
args.bert_tokenizer = BertTokenizer.from_pretrained('./pre_trained_model/BERT/base-uncased')
args.bert_model = BertModel.from_pretrained('./pre_trained_model/BERT/base-uncased')

# bert_large
args.teacher_tokenizer = BertTokenizer.from_pretrained('./pre_trained_model/BERT/base-uncased')
args.teacher_model = BertModel.from_pretrained('./pre_trained_model/BERT/base-uncased')
args.embed_num = len(args.bert_tokenizer.vocab)
# args.bert_optimizer = torch.optim.Adam(params=args.bert_model.parameters(), lr=0.0005, weight_decay=1e-6)

train_data, valid_data, test_data = DataLoader.build_dataset(args)
train_iter = DataLoader.build_iterator(train_data, args)
valid_iter = DataLoader.build_iterator(valid_data, args)
# test_iter = build_iterator_test(test_data, args)
test_iter = DataLoader.build_iterator1(test_data, args)

start = time.time()
# tearcher_model = QNLP_network.Bert(args)
# tearcher_model = QNLP_network.LLM_teacher(args)
base_model = '../models_hf/LLM/' + args.LLM
tokenizer_pth = '../models_hf/LLM/' + args.LLM
lora_weights = '../models_hf/fine-tuning/teacher_model/model_output/' + args.LLM + 'c/' + data_name
# template_path = "../models_hf/fine-tuning/teacher_model/templates/alpaca.json"  # gen
template_path = "../models_hf/fine-tuning/teacher_model/cla/prompt1.json"  # cla

if args.num_epochs != 0:
    tearcher_model = QNLP_network.LLM_teacher(base_model, tokenizer_pth, lora_weights, template_path, args)
init_model = QNLP_network.Embedding(args, args.qbits)

acc_max, p_max, r_max, f1_max = [], [], [], []
model = QNLP_network.RNN(args.hidden_size, args.output_size, args.num_layers, args)
trainable_params = model.trainable_params()
total_params_count = 0
for param in trainable_params:
    shape = param.shape
    param_count = np.prod(shape)
    # if param.name == 'qembedding1.weight' or 'qembedding2.weight':
    #     param_count = param_count * 2
    print(param.name, param_count)
    total_params_count += param_count
print(f"Total trainable parameters count: {total_params_count}")
cache_dir = './save_model'

for i in range(10):
    print("===================================== Iter:", i, "=====================================")

    test_acc_total, test_p_total, test_r_total, test_f1_total = [], [], [], []
    # tokenizer = BasicTokenizer(True)
    # embedding, vocab = Glove.from_pretrained('6B', 100)

    # loss_fn1 = nn.CrossEntropyLoss()
    loss_fn1 = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_fn2 = nn.KLDivLoss(reduction='batchmean')
    loss_fn3 = JSDivLoss()
    # optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr, weight_decay=0.8)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr)
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    best_valid_loss = float('inf')
    best_valid_acc = float('inf')
    if args.alpha != 0:
        init_model.load_state_dict(torch.load('./init_model/' + data_name + '_1.pt'))  # BERT-base

    for epoch in range(args.num_epochs):
        train_one_epoch(model, train_iter, epoch)
        epoch_loss = 0
        valid_loss, valid_acc, valid_p, valid_r, valid_f1 = evaluate(model, valid_iter, loss_fn1, epoch)

        if valid_loss <= best_valid_loss or valid_acc >= best_valid_acc:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            qbits = args.qbits
            ckpt_file_name = os.path.join(cache_dir, f'{data_name}_{qbits}_{epoch+1}.ckpt')
            # ckpt_file_name = os.path.join(cache_dir, f'model_{epoch+1}.ckpt')
            ms.save_checkpoint(model, ckpt_file_name)

    print("----------- test -----------")
    qbits = args.qbits
    test_start = time.time()
    models = []

    files = sorted(os.listdir(cache_dir))
    for name in files:
        if name.endswith('.ckpt'):
            models.append(name)
    model_steps = sorted([int(m.split('_')[-1].split('.')[0]) for m in models])

    for step in model_steps[-5:]:
        # best_model = 'model_{}.ckpt'.format(step)
        best_model = '{}_{}_{}.ckpt'.format(data_name, qbits, step)
        m_path = os.path.join(cache_dir, best_model)
        # print('the {} model is loaded...'.format(m_path))
        param_dict = ms.load_checkpoint(m_path)
        ms.load_param_into_net(model, param_dict)
        test_loss, test_acc, test_p, test_r, test_f1 = evaluate(model, test_iter, loss_fn1)

    # directory_path = "./save_model"
    # for filename in os.listdir(directory_path):
    #     if filename.endswith(".ckpt"):
    #         file_path = os.path.join(directory_path, filename)
    #         try:
    #             os.remove(file_path)
    #             # print(f"{file_path} has been deleted.")
    #         except Exception as e:
    #            print(f"Error deleting {file_path}. Reason: {e}")

    test_end = time.time()
    print(test_end - test_start)
    test_acc_total.append(test_acc)
    test_p_total.append(test_p)
    test_r_total.append(test_r)
    test_f1_total.append(test_f1)
    acc_max.append(max(test_acc_total))
    p_max.append(max(test_p_total))
    r_max.append(max(test_r_total))
    f1_max.append(max(test_f1_total))
print("Acc(mean±std): {:.2f}±{:.2f}".format(np.mean(acc_max) * 100, np.std(acc_max) * 100))
print("P(mean±std): {:.2f}±{:.2f}".format(np.mean(p_max) * 100, np.std(p_max) * 100))
print("R(mean±std): {:.2f}±{:.2f}".format(np.mean(r_max) * 100, np.std(r_max) * 100))
print("F1(mean±std): {:.2f}±{:.2f}".format(np.mean(f1_max) * 100, np.std(f1_max) * 100))
print("----------- time cost -----------")
end = time.time()
print(end - start)
