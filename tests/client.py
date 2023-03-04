import numpy as np
import grpc
import cv2

from polypnet.grpc.detect_pb2 import BatchPolypDetectionRequest, PolypDetectionRequest, Image
from polypnet.grpc.detect_pb2_grpc import PolypDetectionServiceStub


if __name__ == "__main__":
    with grpc.insecure_channel('localhost:12002') as channel:
        stub = PolypDetectionServiceStub(channel)
        with open('tests/data/sample-ugu-2.jpg', 'rb') as f:
            content = f.read()
        requests = [
            PolypDetectionRequest(image=Image(content=content))
            for _ in range(3)
        ]
        batch = BatchPolypDetectionRequest(requests=requests)
        responses = stub.BatchPolypDetect(batch)
        print(responses)

    polys = np.array([
        [p.x, p.y]
        for p in responses.responses[0].polyps[0].boundingPoly
    ])

    polys = polys.reshape((-1, 1, 2))
    img = np.zeros((995, 1280))

    image = cv2.polylines(img, [polys],
        True, (255, 0, 0), 2)

    cv2.imwrite('logs/out.png', image)
