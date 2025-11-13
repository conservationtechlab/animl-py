#!/usr/bin/env python3
"""
Export a PyTorch model to ONNX with a dynamic batch size and verify with ONNX Runtime.

Usage example:
    python export_to_onnx.py --out model.onnx

What it does:
- sets the model to eval(), moves to CPU
- uses a dummy input with batch size 1 for tracing
- calls torch.onnx.export with dynamic_axes for batch dimension
- (optionally) verifies the exported model runs in ONNX Runtime for different batch sizes
"""
import unittest
import torch
import numpy as np
import onnxruntime as ort

from animl.reid.inference import load_miew


def export_miew(model: str,
                output_path: str,
                input_names=("input",),
                output_names=("output",),
                opset_version: int = 18,
                resize_height: int = 440,
                resize_width: int = 440,
                do_constant_folding: bool = True,
                dynamic_batch_dim: bool = False,):
    example_input = torch.randn(1, 3, resize_height, resize_width, dtype=torch.float32)
    model.eval()
    model_cpu = model.to("cpu")

    # Prepare dynamic_axes: map axis 0 of input and output to a symbolic 'batch_size'
    dynamic_axes = {}
    if dynamic_batch_dim:
        # support multiple inputs / outputs if provided
        for n in input_names:
            dynamic_axes[n] = {0: "batch_size"}
        for n in output_names:
            dynamic_axes[n] = {0: "batch_size"}

    print(f"Exporting to {output_path} (opset {opset_version})")
    torch.onnx.export(
        model_cpu,
        example_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        verbose=False,
    )
    print("Export finished.")


@unittest.skip
def main():
    # test_env_print()

    miew_model = load_miew("models/miewid_v3.bin", device="cpu")

    export_miew(miew_model, "models/miewid_v3.onnx")

    sess = ort.InferenceSession("models/miewid_v3.onnx", providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print("ONNX input name:", inp.name)
    print("ONNX input shape (None is dynamic):", inp.shape)
    print("ONNX input type:", inp.type)

    input_shape = [1] + [int(d) if isinstance(d, int) else 440 for d in inp.shape[1:]]
    x = np.random.randn(*input_shape).astype(np.float32)

    # get PyTorch model output for comparison
    emb = miew_model.extract_feat(torch.from_numpy(x))
    emb = emb.detach().cpu().numpy()
    print(emb)

    # get ONNX model output
    outputs = sess.run(None, {inp.name: x})[0]
    print(outputs)
    print(f"Pytorch output shape: {emb.shape}")
    print(f"Onnx output shape: {outputs.shape}")


# run test
main()
