import torch

from causal_nf.sem_equations.sem_base import SEM


class Chain2(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        if sem_name == "linear":
            functions = [
                lambda u1: u1,
                lambda x1, u2: 10 * x1 - u2,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: (10 * x1 - x2),
            ]
        elif sem_name == "linear-functional":
            functions = [
                lambda u1: u1,
                lambda x1, u1: 10 * x1 - 2 * u1,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: (10 * x1 - x2) / 2,
            ]
        elif sem_name == "non-linear":
            functions = [
                lambda u1: u1,
                lambda x1, u2: torch.exp(x1 / 2.0) + u2 / 4.0,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: 4.0 * (x2 - torch.exp(x1 / 2.0)),
            ]
        elif sem_name == "non-linear-2":
            functions = [
                lambda u1: torch.sigmoid(u1),
                lambda x1, u2: 10 * x1**0.5 - u2,
            ]
            inverses = [
                lambda x1: torch.logit(x1),
                lambda x1, x2: (10 * x1**0.5 - x2),
            ]

        elif sem_name == "non-linear-3":

            functions = [
                lambda u1: u1,
                lambda x1, u2: 1 * x1**2.0 - u2,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: (1 * x1**2.0 - x2),
            ]
        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((2, 2))

        adj[0, :] = torch.tensor([0, 0])
        adj[1, :] = torch.tensor([1, 0])
        if add_diag:
            adj += torch.eye(2)

        return adj

    def intervention_index_list(self):
        return [0]
