import torch

from causal_nf.sem_equations.sem_base import SEM


class ChainConfounded(SEM):
    """ We need to write an adjacency matrix. So in order not to write a new one for each SEM where the 
    confounder affects a different subset of the observed variables, we will just multiply the not needed
    effect by 0 in the full SEM, i.e. instead of not having x_i = ... + x_0, we would have x_i = ... + 0 * x_0
    """
    def __init__(self, sem_name):
        functions = None
        inverses = None
        if sem_name == "linear":
            functions = [
                lambda u1: u1,
                lambda x1, u2: x1 / 4.0 + u2,
                lambda x1, x2, u3: x1 / 3.0 + 10 * x2 - u3,
                lambda x1, x2, x3, u4: 0.2 * x1 + 0.25 * x3 + 2 * u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2 - x1 / 4.0,
                lambda x1, x2, x3: (x1 / 3.0 + 10 * x2 - x3),
                lambda x1, x2, x3, x4: (x4 - 0.25 * x3 - 0.2 * x1) / 2,
            ]
        elif sem_name == "linear-ones":
            functions = [
                lambda u1: u1,
                lambda x1, u2: x1 + u2,
                lambda x1, x2, u3: x1 + x2 - u3,
                lambda x1, x2, x3, u4: x1 + x3 + u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2 - x1,
                lambda x1, x2, x3: x1 + x2 - x3,
                lambda x1, x2, x3, x4: x4 - x3 - x1,
            ]
        elif sem_name == "non-linear-additive-0123":
            functions = [
                lambda u1: u1,
                lambda x1, u2: x1 ** 2 + u2,
                lambda x1, x2, u3: torch.exp(torch.abs(x1)) + torch.exp(x2 / 2.0) + u3 / 4.0,
                lambda x1, x2, x3, u4: x1 ** 4 + (x3 - 5) ** 3 / 15.0 + u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2 - x1 **2,
                lambda x1, x2, x3: 4.0 * (x3 - torch.exp(x2 / 2.0) - torch.exp(torch.abs(x1))),
                lambda x1, x2, x3, x4: x4 - (x3 - 5) ** 3 / 15.0 - x1 ** 4,
            ]
        elif sem_name == "non-linear-additive-012":
            functions = [
                lambda u1: u1,
                lambda x1, u2: x1 ** 2 + u2,
                lambda x1, x2, u3: torch.exp(torch.abs(x1)) + torch.exp(x2 / 2.0) + u3 / 4.0,
                lambda x1, x2, x3, u4: 0 * x1 ** 4 + (x3 - 5) ** 3 / 15.0 + u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2 - x1 **2,
                lambda x1, x2, x3: 4.0 * (x3 - torch.exp(x2 / 2.0) - torch.exp(torch.abs(x1))),
                lambda x1, x2, x3, x4: x4 - (x3 - 5) ** 3 / 15.0 - 0 * x1 ** 4,
            ]
        elif sem_name == "non-linear-additive-013":
            functions = [
                lambda u1: u1,
                lambda x1, u2: x1 ** 2 + u2,
                lambda x1, x2, u3: 0 * torch.exp(torch.abs(x1)) + torch.exp(x2 / 2.0) + u3 / 4.0,
                lambda x1, x2, x3, u4: x1 ** 4 + (x3 - 5) ** 3 / 15.0 + u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2 - x1 **2,
                lambda x1, x2, x3: 4.0 * (x3 - torch.exp(x2 / 2.0) - 0 * torch.exp(torch.abs(x1))),
                lambda x1, x2, x3, x4: x4 - (x3 - 5) ** 3 / 15.0 - x1 ** 4,
            ]
        elif sem_name == "non-linear-additive-023":
            functions = [
                lambda u1: u1,
                lambda x1, u2: 0 * x1 ** 2 + u2,
                lambda x1, x2, u3: torch.exp(torch.abs(x1)) + torch.exp(x2 / 2.0) + u3 / 4.0,
                lambda x1, x2, x3, u4: x1 ** 4 + (x3 - 5) ** 3 / 15.0 + u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2 - 0 * x1 **2,
                lambda x1, x2, x3: 4.0 * (x3 - torch.exp(x2 / 2.0) - torch.exp(torch.abs(x1))),
                lambda x1, x2, x3, x4: x4 - (x3 - 5) ** 3 / 15.0 - x1 ** 4,
            ]
        elif sem_name == "non-linear-additive-01":
            functions = [
                lambda u1: u1,
                lambda x1, u2: x1 ** 2 + u2,
                lambda x1, x2, u3: 0 * torch.exp(torch.abs(x1)) + torch.exp(x2 / 2.0) + u3 / 4.0,
                lambda x1, x2, x3, u4: 0 * x1 ** 4 + (x3 - 5) ** 3 / 15.0 + u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2 - x1 **2,
                lambda x1, x2, x3: 4.0 * (x3 - torch.exp(x2 / 2.0) - 0 * torch.exp(torch.abs(x1))),
                lambda x1, x2, x3, x4: x4 - (x3 - 5) ** 3 / 15.0 - 0 * x1 ** 4,
            ]
        elif sem_name == "non-linear-additive-02":
            functions = [
                lambda u1: u1,
                lambda x1, u2: 0 * x1 ** 2 + u2,
                lambda x1, x2, u3: torch.exp(torch.abs(x1)) + torch.exp(x2 / 2.0) + u3 / 4.0,
                lambda x1, x2, x3, u4: 0 * x1 ** 4 + (x3 - 5) ** 3 / 15.0 + u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2 - 0 * x1 **2,
                lambda x1, x2, x3: 4.0 * (x3 - torch.exp(x2 / 2.0) - torch.exp(torch.abs(x1))),
                lambda x1, x2, x3, x4: x4 - (x3 - 5) ** 3 / 15.0 - 0 * x1 ** 4,
            ]
        elif sem_name == "non-linear-additive-03":
            functions = [
                lambda u1: u1,
                lambda x1, u2: 0 * x1 ** 2 + u2,
                lambda x1, x2, u3: 0 * torch.exp(torch.abs(x1)) + torch.exp(x2 / 2.0) + u3 / 4.0,
                lambda x1, x2, x3, u4: x1 ** 4 + (x3 - 5) ** 3 / 15.0 + u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2 - 0 * x1 **2,
                lambda x1, x2, x3: 4.0 * (x3 - torch.exp(x2 / 2.0) - 0 * torch.exp(torch.abs(x1))),
                lambda x1, x2, x3, x4: x4 - (x3 - 5) ** 3 / 15.0 - x1 ** 4,
            ]
        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((4, 4))

        adj[0, :] = torch.tensor([0, 0, 0, 0])
        adj[1, :] = torch.tensor([1, 0, 0, 0])
        adj[2, :] = torch.tensor([1, 1, 0, 0])
        adj[3, :] = torch.tensor([1, 0, 1, 0])
        if add_diag:
            adj += torch.eye(4)

        return adj

    def intervention_index_list(self):
        return [0, 1, 2, 3]
