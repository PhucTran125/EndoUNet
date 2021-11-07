import cv2
import numpy as np

from loguru import logger
from typing import List

from polypnet.grpc.detect_pb2 import Polyp, Point


def polyps_from_mask(mask, orig_shape, input_shape, min_size, threshold) -> List[Polyp]:
    coef_y = orig_shape[0] / input_shape[0]
    coef_x = orig_shape[1] / input_shape[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
    mask = cv2.dilate(mask, kernel, iterations=5)
    mask = cv2.erode(mask, kernel, iterations=7)

    _, thresh = cv2.threshold(mask[:, :, 0], 50, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    polyps = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_size:
            logger.debug('too small')
            continue
        tmp = mask.copy()
        separate = np.zeros_like(mask)
        cv2.drawContours(separate, [cnt], -1, 255, -1)
        tmp[separate < 255] = 0
        weight = np.sum(tmp)

        if weight / area <= min_size or np.max(tmp) <= threshold:
            logger.debug('under threshold')
            continue

        # Scale contour to original size
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cnt[:, :, 0] = cnt[:, :, 0] * coef_x
        cnt[:, :, 1] = cnt[:, :,  1] * coef_y

        # Convert contour to object
        bounding_poly = []
        for i in range(cnt.shape[0]):
            coords = cnt[i, 0, :]
            point = Point(x=coords[0], y=coords[1])
            bounding_poly.append(point)

        polyp = Polyp(boundingPoly=bounding_poly, confidence=0.9)
        polyps.append(polyp)

    return polyps
