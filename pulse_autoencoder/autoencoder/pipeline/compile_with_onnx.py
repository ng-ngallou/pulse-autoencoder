"""
Export trained PulseAutoEncoder to ONNX format.

Usage:
    python export_onnx.py
"""

import numpy as np
import onnx
import onnxruntime as ort
import torch

from pulse_autoencoder.autoencoder.model.autoencoder_model import PulseAutoEncoder

MODEL_PATH = "/pulse_autoencoder/autoencoder/model/mixed_data_autoencoder.pth"
OUTPUT_PATH = "../model/pulse_autoencoder.onnx"
INPUT_LENGTH = 4096  # target_len used during training


def main() -> None:
    device = torch.device("cpu")

    model = PulseAutoEncoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 1, INPUT_LENGTH)

    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        input_names=["signal"],
        output_names=["reconstruction"],
        dynamic_axes={
            "signal": {0: "batch_size"},
            "reconstruction": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"Exported to {OUTPUT_PATH}")

    # validate
    onnx_model = onnx.load(OUTPUT_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validated")

    # verify outputs match
    session = ort.InferenceSession(OUTPUT_PATH)
    onnx_out = session.run(None, {"signal": dummy_input.numpy()})[0]

    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    diff = np.abs(torch_out - onnx_out).max()
    print(f"Max absolute difference (PyTorch vs ONNX): {diff:.2e}")


if __name__ == "__main__":
    main()
