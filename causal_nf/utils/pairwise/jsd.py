from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import torch





def jensen_shannon_divergence(p: torch.distributions.Distribution, q: torch.distributions.Distribution) -> torch.tensor:
    m = MultivariateNormal(
        0.5 * (p.loc + q.loc),
        0.5 * (p.covariance_matrix + q.covariance_matrix)
    )
    return 0.5 * (
        kl_divergence(p, m) + \
        kl_divergence(q, m)
    )








