import numpy as np

from polypnet.engine import PolypnetEngine
from polypnet.grpc.detect_pb2 import PolypDetectionRequest, Image

engine = PolypnetEngine('models/polypnet-200116')

with open('tests/data/sample-1.jpg', 'rb') as f:
    img = f.read()

out = engine.predict_polyps([
    PolypDetectionRequest(image=Image(content=img)),
    PolypDetectionRequest(image=Image(content=img))
])
