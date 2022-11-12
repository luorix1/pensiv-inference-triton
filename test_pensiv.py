import asyncio

import numpy as np
import tritonclient.http.aio as httpclient
from tritonclient.utils import InferenceServerException


async def test_infer(triton_client, model_name, model_input):
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput("INPUT__0", list(model_input.shape), "FP32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(model_input, binary_data=True)
    print("input initialized")

    outputs.append(httpclient.InferRequestedOutput("OUTPUT__0", binary_data=True))
    print("output initialized")

    results = await triton_client.infer(model_name, inputs, outputs=outputs)

    return results


async def main():
    frames = np.load("frames.npy")
    mels = np.load("mels.npy")

    print("numpy files loaded")
    print(frames.shape)

    try:
        triton_client = httpclient.InferenceServerClient(
            url="localhost:8000", verbose=False
        )
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    print("triton client initialized")

    vision_model_name = "pensiv_vision"
    audio_model_name = "pensiv_audio"
    # lstm_model_name = "pensiv_lstm"

    # statistics = await triton_client.get_inference_statistics(model_name=vision_model_name, headers=None)
    # print(statistics)

    # results = await test_infer(
    #     triton_client=triton_client,
    #     model_name=vision_model_name,
    #     frames=frames
    # )
    # print(results.get_response())

    vision_result = await test_infer(
        triton_client=triton_client,
        model_name=vision_model_name,
        model_input=frames[:16],
    )
    print(vision_result.get_response())

    vision_result = await test_infer(
        triton_client=triton_client,
        model_name=vision_model_name,
        model_input=frames[16:32],
    )
    print(vision_result.get_response())

    # vision_result, audio_result = await asyncio.gather(
    #     test_infer(
    #         triton_client=triton_client,
    #         model_name=vision_model_name,
    #         model_input=frames[:10]
    #     ),
    #     test_infer(
    #         triton_client=triton_client,
    #         model_name=audio_model_name,
    #         model_input=mels[:10]
    #     )
    # )

    # print(vision_result.get_response())
    # print(audio_result.get_response())

    await triton_client.close()
    print("PASS: infer")


if __name__ == "__main__":
    asyncio.run(main())
