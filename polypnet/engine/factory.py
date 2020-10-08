from loguru import logger

from .tf import TensorflowPolypnetEngine
from .base import IPolypnetEngine
from .onnx_ import OnnxPolypnetEngine

__TYPE_MAP = {
    'tf': TensorflowPolypnetEngine,
    'onnx': OnnxPolypnetEngine
}


def create_polypnet_engine(type: str, **kwargs) -> IPolypnetEngine:
    if type not in __TYPE_MAP.keys():
        raise ValueError(f'Unsupported engine type {type}')

    logger.info(f'Loading {__TYPE_MAP[type].__name__}')
    return __TYPE_MAP[type](**kwargs)
