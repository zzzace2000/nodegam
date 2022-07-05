"""Neural Network related utils like Entmax and Modules."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from torch.jit import script


def to_one_hot(y, depth=None):
    """Make the target become one-hot encoding.

    Takes integer with n dims and converts it to 1-hot representation with n + 1 dims.
    The n+1'st dimension will have zeros everywhere but at y'th index, where it will be equal to 1.

    Args:
        y: Input integer (IntTensor, LongTensor or Variable) of any shape.
        depth (int): The size of the one hot dimension.

    Returns:
        y_onehot: The onehot encoding of y.
    """
    y_flat = y.to(torch.int64).view(-1, 1)
    depth = depth if depth is not None else int(torch.max(y_flat)) + 1
    y_one_hot = torch.zeros(y_flat.size()[0], depth, device=y.device).scatter_(1, y_flat, 1)
    y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))
    return y_one_hot


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """Sparsemax function.

    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.

    By Ben Peters and Vlad Niculae.
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)

        Args:
            input: Any dimension.
            dim: Dimension along which to apply.

        Returns:
            output (Tensor): Same shape as input.
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold.

        Args:
            input: Any dimension.
            dim: Dimension along which to apply the sparsemax.

        Returns:
            The threshold value.
        """

        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size


sparsemax = lambda input, dim=-1: SparsemaxFunction.apply(input, dim)
sparsemoid = lambda input: (0.5 * input + 0.5).clamp_(0, 1)


class Entmax15Function(Function):
    """Entropy Max (EntMax).

    An implementation of exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim

        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val  # same numerical stability trick as for softmax
        input = input / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = torch.clamp(input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)

        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt ** 2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean ** 2)
        delta = (1 - ss) / rho

        # NOTE this is not exactly the same as in reference algo
        # Fortunately it seems the clamped values never wrongly
        # get selected by tau <= sorted_z. Prove this!
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size


class Entmoid15(Function):
    """A highly optimized equivalent of lambda x: Entmax15([x, 0])."""

    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    @script
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        tau = (input + torch.sqrt(F.relu(8 - input ** 2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * F.relu(tau - input, inplace=True) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    @script
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input


entmax15 = lambda input, dim=-1: Entmax15Function.apply(input, dim)
entmoid15 = Entmoid15.apply


def my_one_hot(val, dim=-1):
    """Make one hot encoding along certain dimension and not just the last dimension.

    Args:
        val: A pytorch tensor.
        dim: The dimension.
    """
    max_cls = torch.argmax(val, dim=dim)
    onehot = F.one_hot(max_cls, num_classes=val.shape[dim])

    # swap back the dimension
    if dim != -1 and dim != val.ndim - 1:
        the_dim = list(range(onehot.ndim))
        the_dim.insert(dim, the_dim.pop(-1))
        onehot = onehot.permute(the_dim)

    return onehot


class _Temp(nn.Module):
    """Shared base class to do temperature annealing for EntMax/SoftMax/GumbleMax functions."""

    def __init__(self, steps, max_temp=1., min_temp=0.01, sample_soft=False):
        """Annealing temperature from max to min in log10 space.

        Args:
            steps: The number of steps to change from max_temp to the min_temp in log10 space.
            max_temp: The max (initial) temperature.
            min_temp: The min (final) temperature.
            sample_soft: If False, the model does a hard operation after the specified steps.
        """
        super().__init__()
        self.steps = steps
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.sample_soft = sample_soft

        # Initialize to nn Parameter to store it in the model state_dict
        self.tau = nn.Parameter(torch.tensor(max_temp, dtype=torch.float32), requires_grad=False)

    def forward(self, logits, dim=-1):
        # During training and under annealing, run a soft max operation
        if self.sample_soft or (self.training and self.tau.item() > self.min_temp):
            return self.forward_with_tau(logits, dim=dim)

        # In test time, sample a hard max
        with torch.no_grad():
            return self.discrete_op(logits, dim=dim)

    def discrete_op(self, logits, dim=-1):
        return my_one_hot(logits, dim=dim).float()

    @property
    def is_deterministic(self):
        return (not self.sample_soft) and (not self.training or self.tau.item() <= self.min_temp)

    def temp_step_callback(self, step):
        # Calculate the temp; allow fractional step!
        if step >= self.steps:
            self.tau.data = torch.tensor(self.min_temp, dtype=torch.float32)
        else:
            logmin = np.log10(self.min_temp)
            logmax = np.log10(self.max_temp)
            # Linearly interpolate it;
            logtemp = logmax + step / self.steps * (logmin - logmax)
            temp = (10 ** logtemp)
            self.tau.data = torch.tensor(temp, dtype=torch.float32)

    def forward_with_tau(self, logits, dim):
        raise NotImplementedError()


class SMTemp(_Temp):
    """Softmax with temperature annealing."""
    def forward_with_tau(self, logits, dim):
        return F.softmax(logits / self.tau.item(), dim=dim)


class GSMTemp(_Temp):
    """Gumbel Softmax with temperature annealing."""
    def forward_with_tau(self, logits, dim):
        return F.gumbel_softmax(logits, tau=self.tau.item(), dim=dim)


class EM15Temp(_Temp):
    """EntMax15 with temperature annealing."""
    def forward_with_tau(self, logits, dim):
        return entmax15(logits / self.tau.item(), dim=dim)


class EMoid15Temp(_Temp):
    """Entmoid with temperature annealing."""
    def __init__(self, **kwargs):
        # It always does soft operation.
        kwargs['sample_soft'] = True
        super().__init__(**kwargs)

    def forward_with_tau(self, logits, dim=-1):
        return entmoid15(logits / self.tau.item())

    def discrete_op(self, logits, dim=-1):
        # Do not handle the logits=0 since it's quite rare in opt
        # And I think it's fine to output 0.5
        return torch.sign(logits) / 2 + 0.5


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ModuleWithInit(nn.Module):
    """Base class for pytorch module with data-aware initializer on first batch."""
    def __init__(self):
        super().__init__()
        self._is_initialized_tensor = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self._is_initialized_bool = None
        # Note: this module uses a separate flag self._is_initialized so as to achieve both
        # * persistence: is_initialized is saved alongside model in state_dict
        # * speed: model doesn't need to cache
        # please DO NOT use these flags in child modules
        # I change the type to torch.float32 to use apex 16 precision training

    def initialize(self, *args, **kwargs):
        """initialize module tensors using first batch of data."""
        raise NotImplementedError("Please implement ")

    def __call__(self, *args, **kwargs):
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1
            self._is_initialized_bool = True
        return super().__call__(*args, **kwargs)
