import logging
import logging.config
import os

import tritonclient.grpc.aio as grpcclient

logging.config.fileConfig("logging.conf")
logger = logging.getLogger()


def get_triton_client():
    # set up Triton connection
    TRITONURL = "localhost:8001"
    # TODO check that always available ...
    try:
        triton_client = grpcclient.InferenceServerClient(url=TRITONURL)
        logger.info(f"Server ready? {triton_client.is_server_ready()}")
    except Exception as e:
        logger.error("client creation failed: " + str(e))
    return triton_client
