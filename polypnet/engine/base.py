from abc import ABC, abstractmethod

from polypnet.grpc.detect_pb2 import BatchPolypDetectionRequest, BatchPolypDetectionResponse

class IPolypnetEngine(ABC):
    @abstractmethod
    def predict_polyps(self, batch_request: BatchPolypDetectionRequest) -> BatchPolypDetectionResponse:
        raise NotImplementedError()
