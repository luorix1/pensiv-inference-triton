"""API server that accepts requests and returns values."""
import asyncio
import concurrent.futures
import functools
import logging
import logging.config
import os
import time

import numpy as np

# import tritonclient.http.aio as httpclient
import tritonclient.http as httpclient
from fastapi import FastAPI
from starlette.status import (
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from src.api.model import *
from src.api.predictor import Preprocessor

# define logger
if not os.path.exists("logs"):
    os.mkdir("logs")

logging.config.fileConfig("logging.conf")
logger = logging.getLogger()

# define FastAPI client
app = FastAPI()

# define preprocessor
preprocessor = Preprocessor(logger=logger)

# define triton client
# triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)
triton_client = httpclient.InferenceServerClient(
    url="localhost:8000", verbose=False, concurrency=10
)

# define some constants
HOP_LENGTH = 512
N_UNIT = 128
UNIT_LENGTH = 2 * 1 + 1
SKIP_LENGTH = 3
LOAD_LENGTH = UNIT_LENGTH * N_UNIT * SKIP_LENGTH

RESOLUTION = (640, 360)


@app.get("/")
def healthcheck() -> bool:
    """Check the server's status."""
    return True


@app.get("/health")
async def get_server_health() -> bool:
    """Check the Triton Inference Server's status."""

    if await triton_client.is_server_live():
        logger.info("Server is alive")
        await triton_client.close()
        return {"success": True}
    else:
        await triton_client.close()
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton server not available",
        )


@app.get("/models/status")
async def get_model_repository():
    """Check the Triton Inference Server's loaded models"""

    if await triton_client.get_model_repository_index():
        await triton_client.close()
        return {"success": True}
    else:
        await triton_client.close()
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton server not available",
        )


@app.post("/preprocess")
async def run_preprocess(data: PensivModelRequest):
    """
    Preprocess given video file
    """

    logger.info(f"Received video file at {data.filename}")
    frames, mels = await preprocessor.preprocess(
        id=data.id,
        date=data.date,
        filename=data.filename,
        path=data.path,
    )

    logger.info(f"frames shape: {frames.shape}")
    logger.info(f"mels shape: {mels.shape}")

    np.save("frames.npy", frames)
    np.save("mels.npy", mels)

    return {"prediction": []}


@app.post("/inference")
async def run_inference(data: PensivModelRequest):
    """
    Run Pensiv model (vision + audio + LSTM) on given video file
    """

    start = time.time()
    logger.info(f"Received inference request from ID {data.id}")
    logger.info(f"Received video file {data.filename}")

    frame_list, mel_list, num_requests = await preprocessor.preprocess(
        id=data.id,
        date=data.date,
        filename=data.filename,
        path=data.path,
    )
    # np.save("frames.npy", frame_list[0])
    # np.save("mels.npy", mel_list[0])
    preprocess = time.time()

    logger.info(
        f"{data.filename} preprocessing complete in {preprocess - start} seconds"
    )
    logger.info(f"number of requests to run: {num_requests}")
    logger.info(f"size of frame: {frame_list[0].shape}")
    logger.info(f"size of mel-spectrogram: {mel_list[0].shape}")

    vision_async_requests = []
    audio_async_requests = []

    # collect requests for vision model
    for i in range(num_requests):
        # Initialize the data
        inputs = []
        inputs.append(
            httpclient.InferInput("INPUT__0", list(frame_list[i].shape), "FP32")
        )
        inputs[0].set_data_from_numpy(frame_list[i], binary_data=True)

        # Initialize the output
        outputs = [httpclient.InferRequestedOutput("OUTPUT__0", binary_data=True)]

        vision_async_requests.append(
            triton_client.async_infer(
                model_name="pensiv_vision", inputs=inputs, outputs=outputs
            )
        )

    # collect requests for audio model
    for i in range(num_requests):
        # Initialize the data
        inputs = []
        inputs.append(
            httpclient.InferInput("INPUT__0", list(mel_list[i].shape), "FP32")
        )
        inputs[0].set_data_from_numpy(mel_list[i], binary_data=True)

        # Initialize the output
        outputs = [httpclient.InferRequestedOutput("OUTPUT__0", binary_data=True)]

        audio_async_requests.append(
            triton_client.async_infer(
                model_name="pensiv_audio", inputs=inputs, outputs=outputs
            )
        )

    # run inference with vision and audio model
    mid_features = []
    vision_outputs = []
    audio_outputs = []

    vision_start = time.time()
    for i in range(num_requests):
        # Get the result from the initiated asynchronous inference request.
        # Note the call will block till the server responds.
        vision_result = vision_async_requests[i].get_result()
        vision_output = vision_result.as_numpy("OUTPUT__0")
        vision_outputs.append(vision_output)
    vision_end = time.time()
    logger.info(f"vision inference complete in {vision_end - vision_start} seconds")

    audio_start = time.time()
    for i in range(num_requests):
        # Get the result from the initiated asynchronous inference request.
        # Note the call will block till the server responds.
        audio_result = audio_async_requests[i].get_result()
        audio_output = audio_result.as_numpy("OUTPUT__0")
        audio_outputs.append(audio_output)
    audio_end = time.time()
    logger.info(f"audio inference complete in {audio_end - audio_start} seconds")

    # Reshape features to be compatible as input for LSTM model
    for i in range(num_requests):
        mid_feature = np.concatenate([vision_outputs[i], audio_outputs[i]], axis=-1)
        mid_features.append(mid_feature)

    mid_features = np.concatenate(mid_features, axis=0)[None, :]
    logger.info(mid_features.shape)

    # run inference through LSTM model
    # Initialize the data
    inputs = []
    inputs.append(httpclient.InferInput("INPUT__0", list(mid_features.shape), "FP32"))
    inputs[0].set_data_from_numpy(mid_features, binary_data=True)

    # Initialize the output
    outputs = [httpclient.InferRequestedOutput("OUTPUT__0", binary_data=True)]

    # create async request for LSTM model
    lstm_async_request = triton_client.async_infer(
        model_name="pensiv_lstm", inputs=inputs, outputs=outputs
    )

    # get final output
    result = lstm_async_request.get_result()

    # check completion (for debugging)
    print(result.get_response())

    # parse output
    output = result.as_numpy("OUTPUT__0")[0]
    output = np.exp(output)
    output = output[..., 1] / (output[..., 0] + output[..., 1])
    output = np.log(output / (1 - output))

    return {"prediction": output.tolist()}
