from loguru import logger

from .tf import TensorflowPolypnetEngine
from .base import IPolypnetEngine
from .onnx_ import OnnxPolypnetEngine, OnnxPolypnetEngine2
from .ug import UgUNetEngine
from .duod import DuodUNetEngine

__TYPE_MAP = {
    'tf': TensorflowPolypnetEngine,
    'onnx': OnnxPolypnetEngine,
    'onnx2': OnnxPolypnetEngine2,
    'ugu': UgUNetEngine,
    'duod': DuodUNetEngine,
    'ug': DuodUNetEngine
}


def create_polypnet_engine(type: str, **kwargs) -> IPolypnetEngine:
    if type not in __TYPE_MAP.keys():
        raise ValueError(f'Unsupported engine type {type}')

    logger.info(f'Loading {__TYPE_MAP[type].__name__}')
    return __TYPE_MAP[type](**kwargs)
