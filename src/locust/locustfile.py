"""
This is a Locus file for load testing
"""
from typing import Any

import cv2

from locust import FastHTTPUser, constant, task


class APIUser(FastHTTPUser):
    wait_time = constant(1)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.file_path = "/home/luorix/pensiv-inference-triton/test.mp4"
        self.request = {"file_path": self.file_path}

    @task
    def run_inference(self) -> None:
        """
        Request model inference on given video
        """
        self.client.get("/run_inference", json=self.request)
