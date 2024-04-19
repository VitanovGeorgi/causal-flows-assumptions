import torch.nn.functional as F
import torch


def kl2(x, y):
    kld = torch.nn.KLDivLoss(reduction="batchmean")
    return kld(x, y)

def compute_kl_div(p, q, samples):
    log_p = p.log_prob(samples)
    log_q = q.log_prob(samples)
    return torch.mean(torch.exp(log_p) * (log_p - log_q))