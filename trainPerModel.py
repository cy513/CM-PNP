import utils
import torch
import torch.nn as nn
import models
import numpy as np
import eval
from config import args
import os


def train():
    device = torch.device('cuda' if args.gpu == 1 else 'cpu')
    seq_len_range = [40, 400]

    train_path = './data/{}/train.txt'.format(args.dataset)
    valid_path = './data/{}/valid.txt'.format(args.dataset)
    num_ent, num_rel = utils.get_stat_data('./data/{}/stat.txt'.format(args.dataset))
    train_examples = utils.load_quadruples(train_path)
    valid_examples = utils.load_quadruples(valid_path)
    train_seq_dict = utils.get_obj_seq(train_examples)
    valid_seq_dict = utils.get_obj_seq(valid_examples)

    ent_dict, word_num = utils.entity_to_words('./data/{}/entity2id.txt'.format(args.dataset))
    entattr_model = models.EntAttr_Model(num_ent, num_rel, args.hidden_dim, ent_dict, word_num)
    entattr_model.load_state_dict(
        torch.load('./model/entattr_model_{}.pt'.format(args.dataset), map_location=torch.device(device)))
    entattr_model.to(device)
    entattr_model.eval()

    ent_model = models.Entity_Linear(num_ent, args.hidden_dim)
    ent_model.load_state_dict(
        torch.load('./model/Entity_Linear_{}.pt'.format(args.dataset), map_location=torch.device(device)))
    ent_model.to(device)
    ent_model.eval()

    rel_model = models.Relation_Linear(num_ent, num_rel, args.hidden_dim)
    rel_model.load_state_dict(
        torch.load('./model/Relation_Linear_{}.pt'.format(args.dataset), map_location=torch.device(device)))
    rel_model.to(device)
    rel_model.eval()

    batch_size = 600

    selected_keys = dict()
    for key in train_seq_dict:
        if len(train_seq_dict[key]) > seq_len_range[0] and len(train_seq_dict[key]) < seq_len_range[
            1] and utils.is_available_sequence(train_seq_dict[key]):
            selected_keys[key] = train_seq_dict[key]

    epoch_num = 30
    for key in selected_keys:
        obj_list = selected_keys[key]
        model = models.PerModel(num_ent, num_rel, args.hidden_dim)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
        for epoch in range(epoch_num):
            model.train()
            seq_emb, gt_labels, time_emb = model.get_sequences_and_labels(obj_list, seq_len=int(len(obj_list) / 3))
            n_batch = (len(seq_emb) + batch_size - 1) // batch_size
            remain = len(seq_emb) % batch_size
            for idx in range(n_batch):
                batch_start = idx * batch_size
                batch_end = min(len(seq_emb), (idx + 1) * batch_size)
                if idx == n_batch - 2 and remain < batch_size / 2:
                    batch_end = len(seq_emb)
                y_pred = model(seq_emb[batch_start: batch_end].to(device), time_emb[batch_start: batch_end].to(device),
                               device)
                loss = criterion(y_pred, gt_labels[batch_start: batch_end].to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_end == len(time_emb):
                    break

        m1_mrr, m1_hit_1, m1_hit_3, m1_hit_10 = 0.0, 0.0, 0.0, 0.0
        m2_mrr, m2_hit_1, m2_hit_3, m2_hit_10 = 0.0, 0.0, 0.0, 0.0

        if key not in valid_seq_dict:
            continue
        for item in valid_seq_dict[key]:
            seq, time_emb = model.get_sequence_embed(item[1], selected_keys[key][-31:-1])
            score = model(seq.to(device), time_emb.to(device), device)
            score = score.squeeze(0)
            m1_mrr, m1_hit_1, m1_hit_3, m1_hit_10 = eval.rank(score, item[0], m1_mrr, m1_hit_1, m1_hit_3, m1_hit_10)

            score = utils.get_score_from_models(entattr_model, ent_model, rel_model, np.asarray([[key[0], key[1]]]), device)
            score = score.squeeze(0)
            m2_mrr, m2_hit_1, m2_hit_3, m2_hit_10 = eval.rank(score, item[0], m2_mrr, m2_hit_1, m2_hit_3, m2_hit_10)

        if m1_mrr / len(valid_seq_dict[key]) > (m2_mrr / len(valid_seq_dict[key]) + 0.1):
            model_dir = './periodic_model_{}/'.format(args.dataset)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_name = model_dir+str(key[0])+'_'+str(key[1])+'_periodic_model.pt'
            torch.save(model.state_dict(), model_name)










