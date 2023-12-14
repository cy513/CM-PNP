import torch
import torch.nn as nn


class Entity_Linear(nn.Module):
    def __init__(self, num_ent, hidden_dim):
        super(Entity_Linear, self).__init__()
        self.ent_emb = nn.Parameter(torch.Tensor(num_ent, hidden_dim))
        self.linear = nn.Linear(hidden_dim, num_ent)
        nn.init.xavier_uniform_(self.ent_emb, gain=nn.init.calculate_gain('relu'))

    def forward(self, batch_data):
        ent_emb = self.ent_emb[batch_data[:, 0]]
        out = self.linear(ent_emb)
        return out

class Relation_Linear(nn.Module):
    def __init__(self, num_ent, num_rel, hidden_dim):
        super(Relation_Linear, self).__init__()
        self.rel_emb = nn.Parameter(torch.Tensor(num_rel, hidden_dim))
        self.linear = nn.Linear(hidden_dim, num_ent)
        nn.init.xavier_uniform_(self.rel_emb, gain=nn.init.calculate_gain('relu'))

    def forward(self, batch_data):
        rel_emb = self.rel_emb[batch_data[:, 1]]
        out = self.linear(rel_emb)
        return out

class EntAttr_Model(nn.Module):
    def __init__(self, num_ent, num_rel, hidden_dim, ent_dict, word_num):
        super(EntAttr_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.rel_num = num_rel
        self.ent_num = num_ent
        self.word_num = word_num

        self.words_embd = nn.Parameter(torch.Tensor(self.word_num, self.hidden_dim))
        self.rel_embed = nn.Parameter(torch.Tensor(self.rel_num, self.hidden_dim))
        self.linear = nn.Linear(self.hidden_dim * 3, self.ent_num)
        self.reset_parameters()

        self.ent_dict = ent_dict

    def get_ent_rel_embed(self, batch_data, device):
        batch_size = len(batch_data)
        word1_ten = torch.Tensor(batch_size, self.hidden_dim).to(device)
        word2_ten = torch.Tensor(batch_size, self.hidden_dim).to(device)

        for i, item in enumerate(batch_data):
            word1_ten[i] = self.words_embd[self.ent_dict[str(item[0])][0]]
            word2_ten[i] = self.words_embd[self.ent_dict[str(item[0])][1]]

        rel_idx = batch_data[:, 1]
        rel = self.rel_embed[rel_idx]

        return word1_ten, word2_ten, rel

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.rel_embed, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.words_embd, gain=nn.init.calculate_gain('relu'))

    def forward(self, batch_data, device):
        ent_embed1, ent_embed2, rel_embed = self.get_ent_rel_embed(batch_data, device)
        m_t = torch.cat((ent_embed1, ent_embed2, rel_embed), dim=1)
        out = self.linear(m_t)
        return out

class PerModel(nn.Module):
    def __init__(self, num_ent, num_rel, hidden_dim):
        super(PerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.rel_num = num_rel
        self.ent_num = num_ent

        self.ent_embed = nn.Parameter(torch.Tensor(self.ent_num, self.hidden_dim))
        self.rel_embed = nn.Parameter(torch.Tensor(self.rel_num, self.hidden_dim))
        self.tim_embed = nn.Parameter(torch.Tensor(1, self.hidden_dim))
        nn.init.xavier_uniform_(self.ent_embed, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embed, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.tim_embed, gain=nn.init.calculate_gain('relu'))

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim * 2, self.ent_num)

    def get_sequences_and_labels(self, obj_list, seq_len=30, max_dist=60):

        tim_emb = torch.Tensor(self.tim_embed.cpu().detach().numpy().reshape(self.hidden_dim))
        batch = []
        for item in obj_list:
            batch.append(self.ent_embed[item[0]].unsqueeze(0))

        batches = []
        bat_labels = []
        tim_li = []
        for i in range(len(batch) - seq_len):
            seq_li = torch.cat(batch[i: i + seq_len], dim=0).unsqueeze(0)
            for j in range(i, min(i + max_dist, len(batch) - seq_len - i)):
                batches.append(seq_li)
                time_intev = max(1, obj_list[j + seq_len][1] - obj_list[i + seq_len - 1][1])
                bat_labels.append(torch.LongTensor([obj_list[j + seq_len][0]]))
                tim_li.append((tim_emb * time_intev).unsqueeze(0))
        seq_emb = torch.cat(batches, dim=0)
        gt_labels = torch.cat(bat_labels, dim=0)
        time_emb = torch.cat(tim_li, dim=0)
        return seq_emb, gt_labels, time_emb

    def get_sequence_embed(self, time, obj_list):
        tim_emb = torch.Tensor(self.tim_embed.cpu().detach().numpy().reshape(self.hidden_dim))

        time_intev = max(1, time - obj_list[-1][1])
        batch = []
        for item in obj_list:
            batch.append(self.ent_embed[item[0]].unsqueeze(0))
        batches = torch.cat(batch, dim=0).unsqueeze(0)
        time_emb = (tim_emb * time_intev).unsqueeze(0)
        return batches, time_emb

    def forward(self, seq_emb, join_emb, device):

        hs_lstm = torch.zeros(1, seq_emb.size(0), self.hidden_dim).to(device)
        cs_lstm = torch.zeros(1, seq_emb.size(0), self.hidden_dim).to(device)

        out, (hs_lstm, cs_lstm) = self.lstm(seq_emb, (hs_lstm, cs_lstm))
        hs_lstm = hs_lstm.view(-1, self.hidden_dim)
        output = self.linear(torch.cat((hs_lstm, join_emb), dim=1))
        return output