import utils
import numpy as np
import torch
import eval
import random
import torch.nn as nn
import models
from config import args
import trainPerModel


device = torch.device('cuda' if args.gpu == 1 else 'cpu')
batch_size = 1024

train_examples = utils.load_quadruples('./data/{}/train.txt'.format(args.dataset))
valid_examples = utils.load_quadruples('./data/{}/valid.txt'.format(args.dataset))

num_ent, num_rel = utils.get_stat_data('./data/{}/stat.txt'.format(args.dataset))
valid_array_data = np.asarray(valid_examples)
valid_num = len(valid_array_data)
valid_n_batch = (valid_num + batch_size - 1) // batch_size
train_array_data = np.asarray(train_examples)
train_num = len(train_array_data)
train_n_batch = (train_num + batch_size - 1) // batch_size

rd_idx = [_ for _ in range(train_num)]
random.shuffle(rd_idx)

ent_dict, word_num = utils.entity_to_words('./data/{}/entity2id.txt'.format(args.dataset))
entattr_model = models.EntAttr_Model(num_ent, num_rel, args.hidden_dim, ent_dict, word_num)
entattr_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(entattr_model.parameters(), lr=0.001)
optimizer.zero_grad()

print('start train models...')

epoch_num = 8
best_mrr = 0
for epoch in range(epoch_num):
    entattr_model.train()
    for idx in range(train_n_batch):
        optimizer.zero_grad()
        batch_start = idx * batch_size
        batch_end = min(train_num, (idx + 1) * batch_size)
        train_batch_data = train_array_data[rd_idx[batch_start: batch_end]]
        labels = torch.LongTensor(train_batch_data[:, 2])
        score = entattr_model(train_batch_data, device)
        loss = criterion(score, labels.to(device))
        loss.backward()
        optimizer.step()

    entattr_model.eval()
    mrr, hits1, hits3, hits10 = 0, 0, 0, 0
    for idx in range(valid_n_batch):
        batch_start = idx * batch_size
        batch_end = min(valid_num, (idx + 1) * batch_size)
        batch_data = valid_array_data[batch_start: batch_end]
        labels = torch.LongTensor(batch_data[:, 2])
        score = entattr_model(batch_data, device)
        tim_mrr, tim_hits1, tim_hits3, tim_hits10 = eval.calc_raw_mrr(score, labels.to(device),
                                                                            hits=[1, 3, 10])
        mrr += tim_mrr * len(batch_data)
        hits1 += tim_hits1 * len(batch_data)
        hits3 += tim_hits3 * len(batch_data)
        hits10 += tim_hits10 * len(batch_data)

    mrr = mrr / valid_array_data.shape[0]
    if mrr > best_mrr:
        best_mrr = mrr
        #print('epoch:{}, valid_mrr={}, Loss: {:.6f}'.format(epoch + 1, mrr, loss.item()))
        torch.save(entattr_model.state_dict(), './model/entattr_model_{}.pt'.format(args.dataset))


model_ent = models.Entity_Linear(num_ent, args.hidden_dim)
model_ent.to(device)
optimizer = torch.optim.Adam(model_ent.parameters(), lr=0.001)
best_mrr = 0
epoch_num = 12
for epoch in range(epoch_num):
    model_ent.train()
    for i in range(train_n_batch):
        optimizer.zero_grad()
        batch_start = i * batch_size
        batch_end = min(train_num, (i + 1) * batch_size)
        train_batch_data = train_array_data[batch_start: batch_end]
        labels = torch.LongTensor(train_batch_data[:, 2])
        score = model_ent(train_batch_data)
        loss = criterion(score, labels.to(device))
        loss.backward()
        optimizer.step()

    model_ent.eval()
    mrr, hits1, hits3, hits10 = 0, 0, 0, 0
    for idx in range(valid_n_batch):
        batch_start = idx * batch_size
        batch_end = min(valid_num, (idx + 1) * batch_size)
        batch_data = valid_array_data[batch_start: batch_end]
        labels = torch.LongTensor(batch_data[:, 2])
        score = model_ent(batch_data)
        tim_mrr, tim_hits1, tim_hits3, tim_hits10 = eval.calc_raw_mrr(score, labels.to(device),
                                                                            hits=[1, 3, 10])
        mrr += tim_mrr * len(batch_data)
        hits1 += tim_hits1 * len(batch_data)
        hits3 += tim_hits3 * len(batch_data)
        hits10 += tim_hits10 * len(batch_data)

    mrr = mrr / valid_array_data.shape[0]
    if mrr > best_mrr:
        best_mrr = mrr
        #print('epoch:{}, valid_mrr={}, Loss: {:.6f}'.format(epoch + 1, mrr, loss.item()))
        torch.save(model_ent.state_dict(), './model/Entity_Linear_{}.pt'.format(args.dataset))


model_rel = models.Relation_Linear(num_ent, num_rel, args.hidden_dim)
model_rel.to(device)
optimizer = torch.optim.Adam(model_rel.parameters(), lr=args.lr)
best_mrr = 0
epoch_num = 12
for epoch in range(epoch_num):
    model_rel.train()
    for i in range(train_n_batch):
        optimizer.zero_grad()
        batch_start = i * batch_size
        batch_end = min(train_num, (i + 1) * batch_size)
        train_batch_data = train_array_data[batch_start: batch_end]
        labels = torch.LongTensor(train_batch_data[:, 2])
        score = model_rel(train_batch_data)
        loss = criterion(score, labels.to(device))
        loss.backward()
        optimizer.step()

    model_rel.eval()
    mrr, hits1, hits3, hits10 = 0, 0, 0, 0
    for idx in range(valid_n_batch):
        batch_start = idx * batch_size
        batch_end = min(valid_num, (idx + 1) * batch_size)
        batch_data = valid_array_data[batch_start: batch_end]
        labels = torch.LongTensor(batch_data[:, 2])
        score = model_rel(batch_data)
        tim_mrr, tim_hits1, tim_hits3, tim_hits10 = eval.calc_raw_mrr(score, labels.to(device),
                                                                            hits=[1, 3, 10])
        mrr += tim_mrr * len(batch_data)
        hits1 += tim_hits1 * len(batch_data)
        hits3 += tim_hits3 * len(batch_data)
        hits10 += tim_hits10 * len(batch_data)

    mrr = mrr / valid_array_data.shape[0]
    if mrr > best_mrr:
        best_mrr = mrr
        #print('epoch:{}, valid_mrr={}, Loss: {:.6f}'.format(epoch + 1, mrr, loss.item()))
        torch.save(model_rel.state_dict(), './model/Relation_Linear_{}.pt'.format(args.dataset))

if args.comb_model == 1:
    print('start train periodic model')
    trainPerModel.train()

