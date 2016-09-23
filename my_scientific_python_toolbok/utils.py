import os


def data_path(*joins):
    base_path = os.path.dirname(__file__)
    return os.path.join(base_path, 'data', *joins)
