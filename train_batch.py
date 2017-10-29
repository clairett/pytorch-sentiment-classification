import torch
import torch.nn as nn
from torch import optim
import time, random
import os
from data_helper import load_data
from tqdm import tqdm
from lstm import LSTMSentiment
import data_helper
import torchtext.datasets as datasets
from torchtext import data
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


def train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    acc = 0
    for batch in tqdm(train_iter):
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc


def train_epoch(model, train_iter, loss_function, optimizer):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in train_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc


def evaluate(model, data, loss_function, name):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in data:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ': loss %.2f acc %.1f' % (avg_loss, acc*100))
    return acc


EPOCHS = 10
USE_GPU = torch.cuda.is_available()
EMBEDDING_DIM = 300
HIDDEN_DIM = 150

BATCH_SIZE = 5
timestamp = str(int(time.time()))
best_dev_acc = 0.0


def load_sst(text_field, label_field, batch_size):
    train, dev, test = data.TabularDataset.splits(path='./data/SST2/', train='train.tsv',
                                                  validation='dev.tsv', test='test.tsv', format='tsv',
                                                  fields=[('text', text_field), ('label', label_field)])
    text_field.build_vocab(train, dev, test)
    label_field.build_vocab(train, dev, test)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),\
        batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=-1)
    return train_iter, dev_iter, test_iter


text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter, test_iter = load_sst(text_field, label_field, BATCH_SIZE)

# text_field.vocab.load_vectors(wv_type='', wv_dim=300)
model = LSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
                      use_gpu=USE_GPU, batch_size=BATCH_SIZE)
if USE_GPU:
    model = model.cuda()

# model.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

best_model = model
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.NLLLoss()

print('Training...')
for epoch in range(EPOCHS):
    avg_loss, acc = train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch)
    tqdm.write('Train: loss %.2f acc %.1f' % (avg_loss, acc*100))
    dev_acc = evaluate(model, dev_iter, loss_function, 'Dev')
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        best_model = model
test_acc = evaluate(best_model, test_iter, loss_function,'Test')


out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
torch.save(best_model.state_dict(), out_dir + '/best_model' + '.pth')
