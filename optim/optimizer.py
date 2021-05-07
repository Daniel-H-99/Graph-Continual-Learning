import math
import torch

from . import register_optimizer


class Optimizer(object):
    def __init__(self, args, params):
        super().__init__()
        self.params = list(params)

    @staticmethod
    def add_args(parser):
        pass

    @property
    def optimizer(self):
        if not hasattr(self, 'optimizer_'):
            raise NotImplementedError
        if not isinstance(self.optimizer_, torch.optim.Optimizer):
            raise ValueError('optimizer_ must be an instance of torch.optim.Optimizer')
        return self.optimizer_

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        self.optimizer.load_state_dict(state_dict)

    def clip_grad_value(self, max_value):
        torch.nn.utils.clip_grad_value_(self.params, max_value)

    def clip_grad_norm(self, max_norm):
        if max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(self.params, max_norm)
        else:
            return math.sqrt(sum(p.grad.data.norm() ** 2 for p in self.params if p.grad is not None))

    def step(self, closure=None):
        self.optimizer.step(closure)

    def zero_grad(self):
        self.optimizer.zero_grad()


@register_optimizer('sgd')
class SGD(Optimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self.optimizer_ = torch.optim.SGD(params, args.lr, args.momentum, weight_decay=args.weight_decay)

    def add_args(parser):
        parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
        parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')


@register_optimizer('adagrad')
class Adagrad(Optimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self.optimizer_ = torch.optim.Adagrad(params, args.lr, weight_decay=args.weight_decay)

    def add_args(parser):
        parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')


@register_optimizer('adadelta')
class Adadelta(Optimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self.optimizer_ = torch.optim.Adadelta(params, args.lr, weight_decay=args.weight_decay)

    def add_args(parser):
        parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')


@register_optimizer('adam')
class Adam(Optimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self.optimizer_ = torch.optim.Adam(
            params, args.lr, betas=(args.beta1, args.beta2), amsgrad=args.amsgrad, weight_decay=args.weight_decay
        )

    def add_args(parser):
        parser.add_argument('--beta1', default=0.9, type=float, help='beta1')
        parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
        parser.add_argument('--amsgrad', action='store_true', help='whether to use AMSGrad')
        parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')


@register_optimizer('adamax')
class Adamax(Optimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self.optimizer_ = torch.optim.Adamax(
            params, args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay
        )

    def add_args(parser):
        parser.add_argument('--beta1', default=0.9, type=float, help='beta1')
        parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
        parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')


@register_optimizer('rmsprop')
class RMSprop(Optimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self.optimizer_ = torch.optim.RMSprop(
            params, args.lr, args.alpha, momentum=args.momentum, weight_decay=args.weight_decay
        )

    def add_args(parser):
        parser.add_argument('--alpha', default=0.99, type=float, help='alpha')
        parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
