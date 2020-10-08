import os
import cv2
import numpy as np
import vdimg
import onnx
import onnxruntime as rt

from typing import List
from loguru import logger

from polypnet.grpc.detect_pb2 import Point, PolypDetectionResponse,\
    Polyp, PolypDetectionRequest, BatchPolypDetectionRequest,\
    BatchPolypDetectionResponse
from .base import IPolypnetEngine
from . import utils


class OnnxPolypnetEngine(IPolypnetEngine):
    def __init__(self, model_dir: str, input_shape=(512, 512),
        min_size=100, threshold=200
    ):
        self.model_dir = model_dir
        self.input_shape = input_shape
        self.min_size = min_size
        self.threshold = threshold

        self.__load_model()

    def __load_model(self):
        model_path = os.path.join(self.model_dir, 'model.onnx')
        onnx.checker.check_model(model_path)

        self.__sess = rt.InferenceSession(model_path)

    def __preprocess(self, images_bytes: List[bytes]):
        images = [vdimg.load_img(x) for x in images_bytes]
        orig_sizes = [x.shape for x in images]

        images = [cv2.resize(x, self.input_shape) for x in images]
        images = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in images]
        images = [x.astype(np.float32) for x in images]
        images = [x / 127.5 - 1 for x in images]
        images = [np.expand_dims(x, axis=0) for x in images]
        return images, orig_sizes

    def predict_polyps(self, batch_request: BatchPolypDetectionRequest) -> BatchPolypDetectionResponse:
        requests = batch_request.requests
        images_bytes = [req.image.content for req in requests]
        images, orig_shapes = self.__preprocess(images_bytes)

        feed = dict([
            (input.name, images[n])
            for n, input in enumerate(self.__sess.get_inputs())
        ])
        masks = self.__sess.run(None, feed)[0]

        masks = np.uint8(masks * 255)
        masks = masks[:, :, :, 0]
        masks = np.stack((masks,) * 3, axis=-1)

        responses = []
        for i in range(masks.shape[0]):
            mask = masks[i, :, :]
            polyps = utils.polyps_from_mask(
                mask.copy(), orig_shapes[i][:2],
                self.input_shape, self.min_size,
                self.threshold
            )
            resp = PolypDetectionResponse(polyps=polyps)
            responses.append(resp)

        return BatchPolypDetectionResponse(responses=responses)
