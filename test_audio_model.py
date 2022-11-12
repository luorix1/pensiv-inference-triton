import numpy as np
import torch

from src.ml.pensiv_model import CNNModel


def main():
    feature_dim = 64
    hidden_dim = 512
    half_unit_length, n_unit, skip_length = 1, 128, 3
    n_lstm_layers = 4
    using_resolution = (640, 360)
    hop_length = 512

    unit_length = 2 * half_unit_length + 1
    load_length = unit_length * n_unit * skip_length

    model_audio = CNNModel(in_channels=2, num_classes=feature_dim)

    ckpt_path = "/data/shorts/admin/model/script/bestmodel.pth.tar"
    pretrained_state = torch.load(ckpt_path)
    model_audio_dict = model_audio.state_dict()
    pretrained_state_audio = {
        k[7:]: v for k, v in pretrained_state["model_audio"].items()
    }
    model_audio_dict.update(pretrained_state_audio)
    model_audio.load_state_dict(model_audio_dict, strict=False)

    print("audio model initialization complete")

    mels = np.load("mels.npy")
    print(mels.shape)
    print("mel-spectrograms loaded")

    partial_mels = mels[:128]
    partial_mels = torch.Tensor(partial_mels)

    output = model_audio(partial_mels)
    print(output.shape)


if __name__ == "__main__":
    main()
