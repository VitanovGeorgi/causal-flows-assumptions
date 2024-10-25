import abc
from typing import Any
import pdb 

import torch
from torch import Tensor
from torch.distributions import Transform, constraints


class CausalTransform(Transform, abc.ABC):
    @abc.abstractmethod
    def intervene(self, index: int, value: float) -> None:
        pass

    @abc.abstractmethod
    def stop_intervening(self, index: int) -> None:
        pass

    @abc.abstractmethod
    def intervening(self) -> bool:
        pass


class ConfoundedCausalEquations(CausalTransform):

    domain = constraints.unit_interval
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, functions, inverses, derivatives=None, hidden_vars=None):
        super(ConfoundedCausalEquations, self).__init__(cache_size=0)
        self.functions = functions
        self.inverses = inverses
        self.derivatives = derivatives
        self.hidden_vars = hidden_vars

        self._interventions = dict()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ConfoundedCausalEquations)
    
    def _expand_dimension(self, x: torch.Tensor) -> torch.Tensor:
        x_aux = []
        idx = 0
        for i in range(len(self.inverses)):
            if self.hidden_vars is not None and i in self.hidden_vars:
                x_aux.append(torch.zeros(x.shape[0],).to(x.device)) # they're on cpu anyway
            else:
                x_aux.append(x[:, idx])
                idx += 1
        _x = torch.stack(x_aux, dim=1)
        return _x


    def _call(self, u: Tensor) -> Tensor:
        """ scm.sample((n,)) calls this function. In case there's an intervention, the self._interventions dict will have the index and value of it,
        so when the if statement is true, it'll do the do(), o.w. it'll pass and just sample from the data normally.
        """
        assert u.shape[1] == len(self.functions)

        x = []
        for i, f in enumerate(self.functions):
            if i in self._interventions:
                x_i = torch.ones_like(u[..., i]) * self._interventions[i]
            else:
                x_i = f(*x[:i], u[..., i])
            x.append(x_i)
        x = torch.stack(x, dim=1)

        return x

    def _inverse(self, x: Tensor) -> Tensor:
        """ We need the x_full for the inverse, because we have those variables in the lambdas/SEM definitions.
        Were we to replace them with 0s, we'd be setting us up for failure if those vars are exp/log in the SEM.
        """
        if self.hidden_vars is None:
            assert x.shape[1] == len(self.inverses)
        else:
            assert x.shape[1] + len(self.hidden_vars) == len(self.inverses)

        u = []
        _x = self._expand_dimension(x)
        for i, g in enumerate(self.inverses):
            """ These are calling on the functions defined in the SEMs, the lambdas there!
            That's why there's i + 1
            """
            u_i = g(*_x[..., : i + 1].unbind(dim=-1))
            u.append(u_i)
        u = torch.stack(u, dim=1)

        _included_vars = [i for i in range(len(self.inverses)) if i not in self.hidden_vars]
        return u[:, _included_vars]

    def log_abs_det_jacobian(self, u: Tensor, x: Tensor) -> Tensor:
        if self.derivatives is None:
            return self._log_abs_det_jacobian_autodiff(u, x)

        logdetjac = []
        for i, g in enumerate(self.derivatives):
            grad_i = g(*x[..., : i + 1].unbind(dim=-1))
            logdetjac.append(torch.log(grad_i.abs()))

        return -torch.stack(logdetjac, dim=-1)

    def _log_abs_det_jacobian_autodiff(
        self, u: Tensor, x: Tensor
    ) -> Tensor:
        _x = self._expand_dimension(x)
        _u = self._expand_dimension(u)
        logdetjac = []
        old_requires_grad = x.requires_grad
        _x.requires_grad_(True)
        for i, g in enumerate(self.inverses):  # u = T(x)
            u_i = g(*_x[..., : i + 1].unbind(dim=-1))
            grad_i = torch.autograd.grad(u_i.sum(), _x)[0][..., i]
            if i in self.hidden_vars:
                continue
            logdetjac.append(torch.log(grad_i.abs()))
        x.requires_grad_(old_requires_grad)
        return -torch.stack(logdetjac, dim=-1)

    def intervene(self, index, value) -> None:
        self._interventions[index] = value # dict where we keep if an int is happening of nor

    def stop_intervening(self, index: int) -> None:
        self._interventions.pop(index)

    @property
    def intervening(self) -> bool:
        return len(self._interventions) > 0


class CausalEquations(CausalTransform):

    domain = constraints.unit_interval
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, functions, inverses, derivatives=None):
        super(CausalEquations, self).__init__(cache_size=0)
        self.functions = functions
        self.inverses = inverses
        self.derivatives = derivatives

        self._interventions = dict()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CausalEquations)

    def _call(self, u: Tensor) -> Tensor:
        """ scm.sample((n,)) calls this function. In case there's an intervention, the self._interventions dict will have the index and value of it,
        so when the if statement is true, it'll do the do(), o.w. it'll pass and just sample from the data normally.
        """
        assert u.shape[1] == len(self.functions)

        x = []
        for i, f in enumerate(self.functions):
            if i in self._interventions:
                x_i = torch.ones_like(u[..., i]) * self._interventions[i]
            else:
                x_i = f(*x[:i], u[..., i])
            x.append(x_i)
        x = torch.stack(x, dim=1)

        return x

    def _inverse(self, x: Tensor) -> Tensor:
        assert x.shape[1] == len(self.inverses)
        # CHANGES!!!
        # if not x.shape[1] == len(self.inverses):
            # x = torch.cat((x[:, 0:2], x[:, 3:]), dim=1)

        u = []
        for i, g in enumerate(self.inverses):
            u_i = g(*x[..., : i + 1].unbind(dim=-1))
            u.append(u_i)
        u = torch.stack(u, dim=1)

        return u

    def log_abs_det_jacobian(self, u: Tensor, x: Tensor) -> Tensor:
        if self.derivatives is None:
            return self._log_abs_det_jacobian_autodiff(u, x)

        logdetjac = []
        for i, g in enumerate(self.derivatives):
            grad_i = g(*x[..., : i + 1].unbind(dim=-1))
            logdetjac.append(torch.log(grad_i.abs()))

        return -torch.stack(logdetjac, dim=-1)

    def _log_abs_det_jacobian_autodiff(
        self, u: Tensor, x: Tensor
    ) -> Tensor:
        logdetjac = []
        old_requires_grad = x.requires_grad
        x.requires_grad_(True)
        for i, g in enumerate(self.inverses):  # u = T(x)
            u_i = g(*x[..., : i + 1].unbind(dim=-1))
            grad_i = torch.autograd.grad(u_i.sum(), x)[0][..., i]
            logdetjac.append(torch.log(grad_i.abs()))
        x.requires_grad_(old_requires_grad)
        return -torch.stack(logdetjac, dim=-1)

    def intervene(self, index, value) -> None:
        self._interventions[index] = value # dict where we keep if an int is happening of nor

    def stop_intervening(self, index: int) -> None:
        self._interventions.pop(index)

    @property
    def intervening(self) -> bool:
        return len(self._interventions) > 0
