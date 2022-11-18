"""Async Triton predictor."""
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
import tritonclient.http.aio as httpclient
from moviepy.editor import VideoFileClip


class Preprocessor:
    """Preprocessor for input video"""

    def __init__(self, logger) -> None:
        self.logger = logger

        # set up constant variables
        self.HOP_LENGTH = 512
        self.N_UNIT = 128
        self.UNIT_LENGTH = 2 * 1 + 1
        self.SKIP_LENGTH = 3
        self.LOAD_LENGTH = self.UNIT_LENGTH * self.N_UNIT * self.SKIP_LENGTH

        self.RESOLUTION = (640, 360)

    async def preprocess(self, id: int, date: str, filename: str, path: str):
        # read input video info
        parent = Path(path)
        file_path = parent.joinpath(filename)
        clip = VideoFileClip(str(file_path))
        capture = cv2.VideoCapture(str(file_path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        n_total_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        # generate and load audio-only version of input video
        audio_path = parent.joinpath(f"{str(file_path.stem)}.mp3")
        clip.audio.write_audiofile(str(audio_path))
        data, sample_rate = torchaudio.load(str(audio_path))
        audio_path.unlink()

        # define torchaudio transforms
        pow_to_db = torchaudio.transforms.AmplitudeToDB()
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=4 * self.HOP_LENGTH,
            win_length=2 * self.HOP_LENGTH,
            hop_length=self.HOP_LENGTH,
            pad=0,
            n_mels=128,
        )

        # preprocess raw audio data
        spec = mel_spectrogram(data)
        spec = pow_to_db(spec)
        mps = sample_rate / self.HOP_LENGTH
        mel_length = int((self.UNIT_LENGTH * self.SKIP_LENGTH) * 2 * mps / fps)

        # define number of requests to be made
        num_requests = (
            n_total_frame // (self.UNIT_LENGTH * self.SKIP_LENGTH) - 1
        ) // self.N_UNIT + 1

        # iterate over slices of input video & audio
        aggregated_frames = []
        aggregated_mels = []
        aggregated_features = []

        for index in range(num_requests):
            capture = cv2.VideoCapture(str(file_path))
            w_origin, h_origin = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
            w_new, h_new = min(
                self.RESOLUTION[0], int(self.RESOLUTION[1] * w_origin / h_origin)
            ), min(self.RESOLUTION[1], int(self.RESOLUTION[0] * h_origin / w_origin))
            w_pad = self.RESOLUTION[0] - w_new
            h_pad = self.RESOLUTION[1] - h_new

            frames = []
            start_index = index * self.LOAD_LENGTH
            capture.set(1, start_index)
            for frame_id in range(min(self.LOAD_LENGTH, n_total_frame - start_index)):
                _, frame = capture.read()
                if frame_id % self.SKIP_LENGTH != 0:
                    continue
                frame = cv2.resize(frame, (w_new, h_new))
                frame = np.pad(
                    frame,
                    (
                        (h_pad // 2, h_pad - h_pad // 2),
                        (w_pad // 2, w_pad - w_pad // 2),
                        (0, 0),
                    ),
                    mode="constant",
                )
                frames.append((frame.astype(np.float32) / 255).transpose(2, 0, 1))
            capture.release()
            frames = np.stack(frames, axis=0)
            frames = frames[: (frames.shape[0] // self.UNIT_LENGTH) * self.UNIT_LENGTH]
            frames = frames.reshape(-1, 3 * self.UNIT_LENGTH, *frames.shape[2:])
            aggregated_frames.append(frames)

            mels = []
            for frame_id in range(min(self.LOAD_LENGTH, n_total_frame - start_index)):
                if (
                    frame_id % self.SKIP_LENGTH != 0
                    or (frame_id // self.SKIP_LENGTH) % self.UNIT_LENGTH
                    != self.UNIT_LENGTH // 2
                ):
                    continue
                start = int(
                    (start_index + frame_id - self.UNIT_LENGTH * self.SKIP_LENGTH)
                    * mps
                    / fps
                )
                end = start + mel_length
                l_pad = max(0, -start)
                r_pad = max(0, end - spec.shape[-1])
                start = start + l_pad
                end = end - r_pad
                mel = spec[:, :, start:end]
                if r_pad > 0 or l_pad > 0:
                    mel = np.pad(mel, ((0, 0), (0, 0), (l_pad, r_pad)), mode="constant")
                mels.append(mel)
            mels = np.stack(mels, axis=0)
            mels = mels[: frames.shape[0]]
            mels = np.exp(mels / 20)
            aggregated_mels.append(mels)

        return aggregated_frames, aggregated_mels, num_requests


# class VisionModel:
#     """Vision model that returns batched features from input video."""

#     def __init__(self, logger, triton_client) -> None:
#         """Initialize."""
#         self.logger = logger
#         self.triton_client = triton_client
#         self.model_name = "pensiv_vision"
#         self.outputs = [httpclient.InferRequestedOutput("OUTPUT__0", binary_data=True)]

#     async def infer(
#         self,
#         frames,
#     ):
#         """Run inference and extract features from the frames of the video file."""
#         inputs = []
#         inputs.append(httpclient.InferInput("INPUT__0", list(frames.shape), "FP32"))

#         # Initialize the data
#         inputs[0].set_data_from_numpy(frames, binary_data=True)

#         results = await self.triton_client.infer(
#             self.model_name, inputs, outputs=self.outputs
#         )
#         output = results.as_numpy("OUTPUT__0")

#         return output


# class AudioModel:
#     """Audio model that returns features from preprocessed mel-spectrograms"""

#     def __init__(self, logger, triton_client) -> None:
#         self.logger = logger
#         self.triton_client = triton_client
#         self.model_name = "pensiv_audio"
#         self.outputs = [httpclient.InferRequestedOutput("OUTPUT__1", binary_data=True)]

#     async def infer(self, mels):
#         inputs = []
#         inputs.append(httpclient.InferInput("INPUT__1", list(mels.shape), "FP32"))

#         # Initialize the data
#         inputs[0].set_data_from_numpy(mels, binary_data=True)

#         results = await self.triton_client.infer(
#             self.model_name, inputs, outputs=self.outputs
#         )
#         output = results.as_numpy("OUTPUT__1")

#         return output


# class PensivModel:
#     """Multimodal+LSTM model that returns highlight detection results from input video"""

#     def __init__(self, logger, triton_client) -> None:
#         self.logger = logger
#         self.triton_client = triton_client

#         self.vision_model_name = "pensiv_vision"
#         self.audio_model_name = "pensiv_audio"
#         self.lstm_model_name = "pensiv_lstm"

#         self.vision_outputs = [grpcclient.InferRequestedOutput("OUTPUT__0")]
#         self.audio_outputs = [grpcclient.InferRequestedOutput("OUTPUT__1")]
#         self.lstm_outputs = [grpcclient.InferRequestedOutput("OUTPUT__2")]

#         # set up constant variables
#         self.HOP_LENGTH = 512
#         self.N_UNIT = 128
#         self.UNIT_LENGTH = 2 * 1 + 1
#         self.SKIP_LENGTH = 3
#         self.LOAD_LENGTH = self.UNIT_LENGTH * self.N_UNIT * self.SKIP_LENGTH

#         self.RESOLUTION = (640, 360)

#     async def infer(
#         self,
#         frames,
#         mels,
#         id: int,
#         date: str,
#         filename: str,
#         path: str,
#         threshold: float,
#     ):
#         import time

#         start = time.time()
#         # define vision model inputs
#         vision_inputs = [grpcclient.InferInput("INPUT__0", frames.shape, "FP32")]
#         vision_inputs[0].set_data_from_numpy(frames.astype(np.float32))

#         # # define audio model inputs
#         # audio_inputs = [
#         #     grpcclient.InferInput("INPUT__1", mels.shape, "FP32")
#         # ]
#         # audio_inputs[0].set_data_from_numpy(mels.astype(np.float32))

#         # run inference for vision model
#         vision_results = await self.triton_client.infer(
#             model_name=self.vision_model_name,
#             inputs=vision_inputs,
#             outputs=self.vision_outputs,
#         )
#         vision = time.time()
#         self.logger.info(f"vision inference complete in {vision - start} seconds")

#         # # run inference for audio model
#         # audio_results = await self.triton_client.infer(
#         #     model_name=self.audio_model_name,
#         #     inputs=audio_inputs,
#         #     outputs=self.audio_outputs,
#         # )
#         # audio = time.time()
#         # self.logger.info(f"audio inference complete in {audio - vision} seconds")

#         # concatenate alone batch axis
#         aggregated_features.append(vision_results.as_numpy("OUTPUT__0"))
#         # aggregated_features.append(audio_results.as_numpy("OUTPUT__1"))
#         # aggregated_features = np.concatenate(aggregated_features, axis=0)
#         self.logger.info(
#             f"vision output shape {aggregated_features[0].as_numpy('OUTPUT__0').shape}"
#         )
#         # self.logger.info(f"audio output shape {aggregated_features[1].as_numpy('OUTPUT__1').shape}")

#         return aggregated_features[0]

#         # # define lstm model inputs
#         # lstm_inputs = [
#         #     grpcclient.InferInput("INPUT__2", aggregated_features.shape, "FP32")
#         # ]
#         # lstm_inputs[0].set_data_from_numpy(aggregated_features.astype(np.float32))

#         # # run inference for lstm model
#         # lstm_results = await self.triton_client.infer(
#         #     model_name=self.lstm_model_name,
#         #     inputs=lstm_inputs,
#         #     outputs=self.lstm_outputs,
#         # )

#         # # postprocess output from lstm model
#         # output = lstm_results.as_numpy("OUTPUT__2")
#         # output = np.exp(output)
#         # output = output[..., 1] / (output[..., 0] + output[..., 1])
#         # output = np.log(output / (1 - output))

#         # return output
