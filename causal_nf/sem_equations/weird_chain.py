import torch

from causal_nf.sem_equations.sem_base import SEM


class Weird_chain(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        if sem_name == "linear":
            raise NotImplementedError
        elif sem_name == "non-linear":
            functions = [
                lambda u1: 2*u1,
                lambda _, u2: u2 / 2,
                lambda x1, x2, u3: torch.tanh(x1) + torch.exp(x2) - u3,
                lambda _, x2, x3, u4: torch.log(x2) + torch.abs(x3) + 2*u4/3,
                lambda x1, x2, x3, x4, u5: x4**3 + u5/10,
            ]
            inverses = [
                lambda x1: x1/2,
                lambda _, x2: 2*x2,
                lambda x1, x2, x3: torch.tanh(x1) + torch.exp(x2) - x3,
                lambda x1, x2, x3, x4: 3*(- torch.log(x2) - torch.abs(x3) + x4)/2,
                lambda x1, x2, x3, x4, x5: 10*(x5 - x4**3),
            ]
        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((5, 5))
        # who's causing/parenting who
        adj[0, :] = torch.tensor([0, 0, 0, 0, 0])
        adj[1, :] = torch.tensor([0, 0, 0, 0, 0])
        adj[2, :] = torch.tensor([1, 1, 0, 0, 0])
        adj[3, :] = torch.tensor([0, 1, 1, 0, 0])
        adj[4, :] = torch.tensor([0, 0, 0, 1, 0])
        if add_diag:
            adj += torch.eye(5)

        return adj

    def intervention_index_list(self):
        return [0, 1, 2, 3]
