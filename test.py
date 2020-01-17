import numpy as np

from PIL import Image as PilImage, ImageDraw

from polypnet.engine import PolypnetEngine
from polypnet.grpc.detect_pb2 import PolypDetectionRequest, Image

engine = PolypnetEngine('models/polypnet-200116')

with open('tests/data/sample-1.jpg', 'rb') as f:
    img = f.read()

out = engine.predict_polyps([
    PolypDetectionRequest(image=Image(content=img)),
    PolypDetectionRequest(image=Image(content=img))
])

polyps = out[0].polyps
img = PilImage.open('tests/data/sample-1.jpg')
draw = ImageDraw.Draw(img)

for polyp in polyps:
    num_points = len(polyp.boundingPoly)
    for i, p1 in enumerate(polyp.boundingPoly):
        p2 = polyp.boundingPoly[(i + 1) % num_points]
        draw.line((p1.x, p1.y, p2.x, p2.y), fill=(68, 112, 184), width=4)

img.save('tests/polyps.jpg')
