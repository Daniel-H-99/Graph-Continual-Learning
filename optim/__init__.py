import importlib
import os


OPTIMIZER_REGISTRY = {}


def build_optimizer(args, params):
    params = list(filter(lambda p: p.requires_grad, params))
    return OPTIMIZER_REGISTRY[args.optimizer](args, params)


def register_optimizer(name):
    """Decorator to register a new optimizer."""
    def register_optimizer_cls(cls):
        if name in OPTIMIZER_REGISTRY:
            raise ValueError('Cannot register duplicate optimizer ({})'.format(name))
        OPTIMIZER_REGISTRY[name] = cls
        return cls

    return register_optimizer_cls


# Automatically import any Python files in the optim directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and file[0].isalpha():
        module = file[:file.find('.py')]
        importlib.import_module('optim.' + module)
