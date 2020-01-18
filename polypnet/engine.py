import os
import cv2
import numpy as np
import segmentation_models as sm
import tensorflow as tf
import vdimg

from polypnet.grpc.detect_pb2 import Point, PolypDetectionResponse, Polyp, PolypDetectionRequest


class PolypnetEngine:
    def __init__(self, model_dir, backbone='efficientnetb4', 
        classes=['polyp'], encoder_freeze=True,
        min_size=100, input_shape=(512, 512),
        threshold=200):
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

    def predict_polyps(self, requests):
        if isinstance(requests, PolypDetectionRequest):
            requests = [requests]
        
        images, orig_shapes = [], []
        for req in requests:
            img = req.image.content
            img = vdimg.load_img(img)
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
            polyps = self.polyps_from_mask(mask.copy(), orig_shapes[i][:2])
            resp = PolypDetectionResponse(polyps=polyps)
            responses.append(resp)

        return responses

    def polyps_from_mask(self, mask, orig_shape):
        coef_y = orig_shape[0] / self.input_shape[0]
        coef_x = orig_shape[1] / self.input_shape[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
        mask = cv2.dilate(mask, kernel, iterations=5)
        mask = cv2.erode(mask, kernel, iterations=5)

        _, thresh = cv2.threshold(mask[:, :, 0], 50, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        import pdb
        polyps = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_size:
                continue
            tmp = mask.copy()
            separate = np.zeros_like(mask)
            cv2.drawContours(separate, [cnt], -1, 255, -1)
            tmp[separate < 255] = 0
            weight = np.sum(tmp)

            if weight / area <= self.min_size or np.max(tmp) <= self.threshold:
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


