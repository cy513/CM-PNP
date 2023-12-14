import torch


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def calc_raw_mrr(score, labels, hits=[]):
    with torch.no_grad():
        ranks = sort_and_rank(score, labels)
        ranks += 1
        mrr = torch.mean(1.0 / ranks.float())
        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()

def rank(score, label, mrr_1, hit_1, hit_3, hit_10):
    with torch.no_grad():

        _, indices = torch.sort(score, dim=0, descending=True)
        indices = torch.nonzero(indices == label)
        ranks = indices[0][0]
        ranks += 1
        mrr = 1.0 / ranks
        hits1 = 1 if ranks <= 1 else 0
        hits3 = 1 if ranks <= 3 else 0
        hits10 = 1 if ranks <= 10 else 0

    return mrr+mrr_1, hits1+hit_1, hits3+hit_3, hits10+hit_10