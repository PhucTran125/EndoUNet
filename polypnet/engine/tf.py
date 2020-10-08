import os
import cv2
import numpy as np
import segmentation_models as sm
import tensorflow as tf
import vdimg

from loguru import logger

from . import utils
from polypnet.grpc.detect_pb2 import Point, PolypDetectionResponse,\
    Polyp, PolypDetectionRequest, BatchPolypDetectionRequest,\
    BatchPolypDetectionResponse
from .base import IPolypnetEngine


class TensorflowPolypnetEngine(IPolypnetEngine):
    def __init__(self, model_dir, backbone='efficientnetb4',
        classes=['polyp'], encoder_freeze=True,
        min_size=100, input_shape=(512, 512),
        threshold=200
    ):
        self.__graph = tf.Graph()
        self.__sess = tf.Session(graph=self.__graph)

        self.threshold = threshold
        self.input_shape = input_shape
        self.min_size = min_size
        self.classes = classes
        self.preprocessor = sm.get_preprocessing(backbone)

        n_classes = 1 if len(classes) == 1 else (len(classes) + 1)
        activation = 'sigmoid' if n_classes == 1 else 'softmax'

        with self.__sess.as_default():
            with self.__graph.as_default():
                self.model = sm.Unet(backbone, classes=n_classes, activation=activation,
                    encoder_freeze=encoder_freeze, weights=os.path.join(model_dir, 'weights.h5'),
                    encoder_weights=None)

    def predict_polyps(self, batch_request: BatchPolypDetectionRequest) -> BatchPolypDetectionResponse:
        requests = batch_request.requests

        images, orig_shapes = [], []
        for req in requests:
            img = vdimg.load_img(req.image.content)
            orig_shapes.append(img.shape)
            img = cv2.resize(img, self.input_shape)
            img = self.preprocessor(img)
            images.append(img)

        images_batch = np.stack(images, axis=0)

        with self.__sess.as_default():
            with self.__graph.as_default():
                masks = self.model.predict(images_batch)

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

