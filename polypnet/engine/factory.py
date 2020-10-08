from .tf import TensorflowPolypnetEngine
from .base import IPolypnetEngine

__TYPE_MAP = {
    'tf': TensorflowPolypnetEngine
}


def create_polypnet_engine(type: str, **kwargs) -> IPolypnetEngine:
    if type not in __TYPE_MAP.keys():
        raise ValueError(f'Unsupported engine type {type}')

    return __TYPE_MAP[type](**kwargs)
