import importlib


def find_dataset_def(dataset_name):
    module_name = 'running.training.datasets_adamvs.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    return getattr(module, "MVSDataset")
