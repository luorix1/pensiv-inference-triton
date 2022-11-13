from pathlib import Path

import torch
import torch.nn

from src.api.ml.pensiv_model import CNNModel, LSTMModel

if __name__ == "__main__":
    model_path = Path("/data/shorts/admin/model/script/bestmodel.pth.tar")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = 64
    hidden_dim = 512
    half_unit_length, n_unit, skip_length = 1, 128, 3
    n_lstm_layers = 4
    using_resolution = (640, 360)
    hop_length = 512

    unit_length = 2 * half_unit_length + 1
    load_length = unit_length * n_unit * skip_length

    model_vision = CNNModel(in_channels=3 * unit_length, num_classes=feature_dim)
    model_audio = CNNModel(in_channels=2, num_classes=feature_dim)
    model_lstm = LSTMModel(
        feature_dim=2 * feature_dim, hidden_dim=hidden_dim, n_lstm_layers=n_lstm_layers
    )

    pretrained_state = torch.load(model_path, map_location="cpu")

    model_vision_dict = model_vision.state_dict()
    pretrained_state_vision = {
        k[7:]: v for k, v in pretrained_state["model_vision"].items()
    }
    model_vision_dict.update(pretrained_state_vision)
    model_vision.load_state_dict(model_vision_dict, strict=False)

    vision_scripted = torch.jit.script(model_vision)
    vision_scripted.save("model_repository/pensiv_vision/1/model.pt")

    model_audio_dict = model_audio.state_dict()
    pretrained_state_audio = {
        k[7:]: v for k, v in pretrained_state["model_audio"].items()
    }
    model_audio_dict.update(pretrained_state_audio)
    model_audio.load_state_dict(model_audio_dict, strict=False)

    audio_scripted = torch.jit.script(model_audio)
    audio_scripted.save("model_repository/pensiv_audio/1/model.pt")

    model_lstm_dict = model_lstm.state_dict()
    pretrained_state_lstm = {
        k[7:]: v for k, v in pretrained_state["model_lstm"].items()
    }
    model_lstm_dict.update(pretrained_state_lstm)
    model_lstm.load_state_dict(model_lstm_dict, strict=False)
    lstm_scripted = torch.jit.script(model_lstm)
    lstm_scripted.save("model_repository/pensiv_lstm/1/model.pt")
