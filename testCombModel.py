import utils
import torch
import numpy as np
import eval
import os
from config import args
import models
import utils



device = torch.device('cuda' if args.gpu == 1 else 'cpu')
train_path = './data/{}/train.txt'.format(args.dataset)
test_path = './data/{}/test.txt'.format(args.dataset)
num_ent, num_rel = utils.get_stat_data('./data/{}/stat.txt'.format(args.dataset))
train_examples = utils.load_quadruples(train_path)
test_examples = utils.load_quadruples(test_path)
train_seq_dict = utils.get_obj_seq(train_examples)
test_seq_dict = utils.get_obj_seq(test_examples)

ent_dict, word_num = utils.entity_to_words('./data/{}/entity2id.txt'.format(args.dataset))
entattr_model = models.EntAttr_Model(num_ent, num_rel, args.hidden_dim, ent_dict, word_num)
entattr_model.load_state_dict(torch.load('./model/entattr_model_{}.pt'.format(args.dataset), map_location=torch.device(device)))
entattr_model.to(device)
entattr_model.eval()

ent_model = models.Entity_Linear(num_ent, args.hidden_dim)
ent_model.load_state_dict(torch.load('./model/Entity_Linear_{}.pt'.format(args.dataset), map_location=torch.device(device)))
ent_model.to(device)
ent_model.eval()

rel_model = models.Relation_Linear(num_ent, num_rel, args.hidden_dim)
rel_model.load_state_dict(torch.load('./model/Relation_Linear_{}.pt'.format(args.dataset), map_location=torch.device(device)))
rel_model.to(device)
rel_model.eval()

batch_size = 1024
total_num = len(test_examples)
keydict = {}
model_dir = './periodic_model_{}/'.format(args.dataset)
for i, j, k in os.walk(model_dir):
    for s in k:
        arr = s.split('_')
        key = (int(arr[0]), int(arr[1]))
        if key in test_seq_dict:
            keydict[key] = 1

if len(keydict) == 0 and args.comb_model == 1:
    print('warning: there is not a periodic model')

tmp_examples = test_examples.copy()
for item in tmp_examples:
    if (item[0], item[1]) in keydict:
        test_examples.remove(item)

test_num = len(test_examples)

array_data = np.asarray(test_examples)
n_batch = (test_num + batch_size - 1) // batch_size
mrr, hits1, hits3, hits10 = 0, 0, 0, 0

for idx in range(n_batch):
    batch_start = idx * batch_size
    batch_end = min(test_num, (idx + 1) * batch_size)
    batch_data = array_data[batch_start: batch_end]
    labels = torch.LongTensor(batch_data[:, 2])
    score = utils.get_score_from_models(entattr_model, ent_model, ent_model, batch_data, device)
    tim_mrr, tim_hits1, tim_hits3, tim_hits10 = eval.calc_raw_mrr(score, labels.to(device), hits=[1, 3, 10])
    mrr += tim_mrr * len(batch_data)
    hits1 += tim_hits1 * len(batch_data)
    hits3 += tim_hits3 * len(batch_data)
    hits10 += tim_hits10 * len(batch_data)

np_num = len(array_data)
np_acc_mrr = mrr
np_acc_hits1 = hits1
np_acc_hits3 = hits3
np_acc_hits10 = hits10

acc_mrr, acc_hits1, acc_hits3, acc_hits10 = 0, 0, 0, 0
for key in keydict:
    model_name = model_dir+str(key[0])+'_'+str(key[1])+'_periodic_model.pt'
    periodic_model = models.PerModel(num_ent, num_rel, args.hidden_dim)
    periodic_model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
    periodic_model.to(device)
    periodic_model.eval()

    for item in test_seq_dict[key]:
        seq, time_emb = periodic_model.get_sequence_embed(item[1], train_seq_dict[key][-31:-1])
        score = periodic_model(seq.to(device), time_emb.to(device), device)
        score = score.squeeze(0)
        acc_mrr, acc_hits1, acc_hits3, acc_hits10 = eval.rank(score, item[0], acc_mrr, acc_hits1, acc_hits3, acc_hits10)

mrr = (np_acc_mrr + acc_mrr) / total_num
hits1 = (np_acc_hits1 + acc_hits1) / total_num
hits3 = (np_acc_hits3 + acc_hits3) / total_num
hits10 = (np_acc_hits10 + acc_hits10) / total_num

print("MRR : {:.6f}".format(mrr))
print("Hits @ 1: {:.6f}".format(hits1))
print("Hits @ 3: {:.6f}".format(hits3))
print("Hits @ 10: {:.6f}".format(hits10))
