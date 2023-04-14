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
from . import utils, softmax, sigmoid


class DuodUNetEngine(IPolypnetEngine):
    def __init__(self, model_dir: str, input_shape=(480, 480),
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

        self._sess = rt.InferenceSession(model_path)

    def _preprocess(self, images_bytes: List[bytes]):
        images = [vdimg.load_img(x) for x in images_bytes]
        orig_sizes = [x.shape for x in images]

        images = [cv2.resize(x, tuple(self.input_shape), interpolation=cv2.INTER_LINEAR) for x in images]
        images = [x.astype(np.float32) for x in images]
        images = [x / 255. for x in images]
        images = [np.transpose(x, (2, 0, 1)) for x in images]
        images = [np.expand_dims(x, axis=0) for x in images]
        return images, orig_sizes

    def predict_polyps(self, batch_request: BatchPolypDetectionRequest) -> BatchPolypDetectionResponse:
        requests = batch_request.requests
        images_bytes = [req.image.content for req in requests]
        images, orig_shapes = self._preprocess(images_bytes)

        pos_list = []
        les_list = []
        hp_list = []
        masks_list = []
        for image in images:
            input = self._sess.get_inputs()[0]
            feed = {
                input.name: image
            }
            # mask = self._sess.run([self._sess.get_outputs()[0].name], feed)[0][0]

            infer_result = self._sess.run([], {input_name: img})
            pos = softmax(infer_result[0][0])
            les = softmax(infer_result[1][0])
            hp = sigmoid(infer_result[2][0])
            hp = 1 if hp > 0.5 else 0
            mask = infer_result[3][0]

            pos_list.append(pos)
            les_list.append(les)
            hp_list.append(hp)
            masks_list.append(mask)

        masks = np.stack(masks_list, axis=0)
        masks = masks[:, 0, ...]
        masks = masks >= 0.5
        masks = np.uint8(masks * 255)
        masks = np.stack((masks,) * 3, axis=-1)

        responses = []
        for i in range(masks.shape[0]):
            mask = masks[i, :, :]
            polyps = polyps_from_mask(
                mask.copy(), orig_shapes[i][:2],
                self.input_shape, self.min_size,
                self.threshold
            )
            resp = PolypDetectionResponse(polyps=polyps)
            responses.append(resp)

        return BatchPolypDetectionResponse(responses=responses)


def polyps_from_mask(mask, orig_shape, input_shape, min_size, threshold) -> List[Polyp]:
    coef_y = orig_shape[0] / input_shape[0]
    coef_x = orig_shape[1] / input_shape[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
    mask = cv2.dilate(mask, kernel, iterations=5)
    mask = cv2.erode(mask, kernel, iterations=7)

    _, thresh = cv2.threshold(mask[:, :, 0], 50, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    polyps = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_size:
            logger.debug('too small')
            continue
        tmp = mask.copy()
        tmp = cv2.resize(tmp, orig_shape[::-1])

        # Scale contour to original size
        # M = cv2.moments(cnt)
        # cx = int(M['m10']/M['m00'])
        # cy = int(M['m01']/M['m00'])
        cnt[:, :, 0] = cnt[:, :, 0] * coef_x
        cnt[:, :, 1] = cnt[:, :,  1] * coef_y

        separate = np.zeros(orig_shape)
        cv2.drawContours(separate, [cnt], -1, 255, -1)
        tmp[separate < 255] = 0
        weight = np.sum(tmp)

        if weight / area <= min_size or np.max(tmp) <= threshold:
            logger.debug('under threshold')
            continue

        # Convert contour to object
        bounding_poly = []
        for i in range(cnt.shape[0]):
            coords = cnt[i, 0, :]
            point = Point(x=coords[0], y=coords[1])
            bounding_poly.append(point)

        polyp = Polyp(boundingPoly=bounding_poly, confidence=0.9)
        polyps.append(polyp)

    return polyps
