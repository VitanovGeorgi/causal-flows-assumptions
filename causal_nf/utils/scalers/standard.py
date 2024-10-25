from torch.distributions import Transform, constraints
from torch import Tensor
import torch

class StandardTransform(Transform):
    r"""Creates a transformation :math:`f(x) = \alpha x + \beta`.

    Arguments:
        shift: The shift term :math:`\beta`, with shape :math:`(*,)`.
        log_scale: The unconstrained scale factor :math:`\alpha`, with shape :math:`(*,)`.
        slope: The minimum slope of the transformation.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        shift: Tensor,
        scale: Tensor,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.shift = shift
        self.scale = scale

    def _call(self, x: Tensor) -> Tensor:
        # CHANGES!!!
        # if x.shape[1] != self.shift.shape[0]:
        #     x = torch.cat((x[:, 0:2], x[:, 3:]), dim=1)
        return (x - self.shift) / self.scale

    def _inverse(self, y: Tensor) -> Tensor:
        return y * self.scale + self.shift

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        # CHANGES!!!
        # if x.shape[1] != y.shape[1]:
        #     x = torch.cat((x[:, 0:2], x[:, 3:]), dim=1)
        return -self.scale.abs().log().expand(x.shape)

    def __str__(self):
        my_str = "StandardTransform(shift={}, scale={})".format(self.shift, self.scale)
        return my_str
