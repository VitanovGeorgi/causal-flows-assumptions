import torch

from causal_nf.sem_equations.sem_base import SEM
import numpy as np


class George_Test(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None

        # def s(x):
        #     return torch.log(1.0 + torch.exp(x))

        s = torch.nn.functional.softplus

        """ Variables
                x1  :   Sex
                x2  :   School
                x3  :   Pay            
        """
        if sem_name == "non-linear":
            """ Given x's and u's, find the next x

                """
            functions = [
                lambda u1: u1,  #   find x1
                lambda x1, u2: 0.2 * x1 + s(x1) * u2,    #  given x1, u2, find x2
                lambda x1, x2, u3: 1 * x1 + 2 * x2 + 1 + torch.tanh(u3) #   find x3
            ]
            inverses = [
                lambda x1: x1,  #   find u1
                lambda x1, x2: 1 / s(x2 - 0.2 * x1),  #   given x1, x2, find u2
                lambda x1, x2, x3: torch.atanh(
                    x3 - x1 - 2 * x2
                )  #   find u3
            ]
        elif sem_name == "sym-prod":
            functions = [
                lambda u1: u1,
                lambda x1, u2: 2 * torch.tanh(2 * x1) + 1 / np.sqrt(10) * u2,
                lambda x1, x2, u3: 1 / 2 * x1 * x2 + 1 / np.sqrt(2) * u3
                # lambda x1, x2, x3, u4: torch.tanh(3 / 2 * x1) + np.sqrt(3 / 10) * u4,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: np.sqrt(10) * (x2 - 2 * torch.tanh(2 * x1)),
                lambda x1, x2, x3: np.sqrt(2) * (x3 - 1 / 2 * x1 * x2),
                lambda x1, x2, x3, x4: 1
                / np.sqrt(3 / 10)
                * (x4 - torch.tanh(3 / 2 * x1)),
            ]
        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((3, 3))

        if self.sem_name == "non-linear":
            adj[0, :] = torch.tensor([0, 0, 0])
            adj[1, :] = torch.tensor([1, 0, 0])
            adj[2, :] = torch.tensor([1, 1, 0])
            # adj[3, :] = torch.tensor([0, 0, 1, 0])
        elif self.sem_name == "sym-prod":
            adj[0, :] = torch.tensor([0, 0, 0])  # for x1
            adj[1, :] = torch.tensor([1, 0, 0])
            adj[2, :] = torch.tensor([1, 1, 0])
            # adj[3, :] = torch.tensor([1, 0, 0, 0])
        if add_diag:
            adj += torch.eye(3)

        return adj

    def intervention_index_list(self):
        return [0, 1, 2]
