import grpc

from polypnet.grpc.detect_pb2 import BatchPolypDetectionRequest, PolypDetectionRequest, Image
from polypnet.grpc.detect_pb2_grpc import PolypDetectionServiceStub


if __name__ == "__main__":
    with grpc.insecure_channel('localhost:12001') as channel:
        stub = PolypDetectionServiceStub(channel)
        with open('tests/data/sample-1.jpg', 'rb') as f:
            content = f.read()
        requests = [
            PolypDetectionRequest(image=Image(content=content))
            for _ in range(3)
        ]
        batch = BatchPolypDetectionRequest(requests=requests)
        responses = stub.BatchPolypDetect(batch)
        print(responses)
        print(len(responses.responses))
