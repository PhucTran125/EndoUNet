import json
import grpc
import os
import yaml

from contexttimer import Timer
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor as Executor
from loguru import logger

from polypnet.engine.base import IPolypnetEngine
from polypnet.engine.factory import create_polypnet_engine
from polypnet.grpc.detect_pb2 import BatchPolypDetectionRequest, BatchPolypDetectionResponse
from polypnet.grpc.detect_pb2_grpc import PolypDetectionServiceServicer, add_PolypDetectionServiceServicer_to_server


class PolypDetectionService(PolypDetectionServiceServicer):
    def __init__(self, engine: IPolypnetEngine):
        self.engine = engine

    def BatchPolypDetect(self, request: BatchPolypDetectionRequest, context):
        with Timer() as t:
            responses = self.engine.predict_polyps(request)
        logger.debug(f'Took {t.elapsed:.2f}s')
        return responses

    def StreamPolypDetect(self, request_iterator, context):
        for req in request_iterator:
            batch_request = BatchPolypDetectionRequest(requests=[req])
            response = self.engine.predict_polyps(batch_request).responses[0]
            yield response


def proc_main(configs):
    # Setup logging
    logger.add(
        os.path.join('logs/', 'server.log'),
        rotation='1 days', retention='7 days'
    )

    polypnet_engine = create_polypnet_engine(
        configs['engine']['type'],
        **configs['engine']['kwargs']
    )
    service = PolypDetectionService(polypnet_engine)

    server = grpc.server(Executor(max_workers=configs['concurrency']))
    add_PolypDetectionServiceServicer_to_server(
        service, server
    )
    server.add_insecure_port('[::]:%d' % configs['port'])
    server.start()
    logger.info(f'Serving on port {configs["port"]}')
    server.wait_for_termination()


if __name__ == "__main__":
    with open('configs/config.yml', 'rt') as f:
        configs = yaml.full_load(f)

    procs = []
    for _ in range(configs['workers']):
        p = Process(target=proc_main, args=(configs,))
        p.start()
        procs.append(p)

    # Wait for completion
    for p in procs:
        try:
            p.join()
        except:
            p.terminate()
