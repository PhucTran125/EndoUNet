from concurrent.futures import ThreadPoolExecutor as Executor
from loguru import logger

import json
import grpc

from polypnet.engine import PolypnetEngine
from polypnet.grpc.detect_pb2 import BatchPolypDetectionRequest, BatchPolypDetectionResponse
from polypnet.grpc.detect_pb2_grpc import PolypDetectionServiceServicer, add_PolypDetectionServiceServicer_to_server


class PolypDetectionService(PolypDetectionServiceServicer):
    def __init__(self, configs):
        self.engine = PolypnetEngine(**configs['engine'])

    def BatchPolypDetect(self, request: BatchPolypDetectionRequest, context):
        responses = self.engine.predict_polyps(request.requests)
        return BatchPolypDetectionResponse(responses=responses)

    def StreamPolypDetect(self, request_iterator, context):
        for req in request_iterator:
            response = self.engine.predict_polyps(req)[0]
            yield response


if __name__ == "__main__":
    with open('configs/config.json', 'rt') as f:
        configs = json.load(f)

    server = grpc.server(Executor(max_workers=configs['concurrency']))
    add_PolypDetectionServiceServicer_to_server(
        PolypDetectionService(configs), server
    )
    server.add_insecure_port('[::]:%d' % configs['port'])
    server.start()
    logger.info(f'Serving on port {configs["port"]}')
    server.wait_for_termination()
