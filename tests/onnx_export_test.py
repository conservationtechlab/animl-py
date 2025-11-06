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
import json
import onnxruntime as ort

from animl import load_classifier, load_class_list

@unittest.skip
def export_model(
    model: str,
    output_path: str,
    input_names=("input",),
    output_names=("output",),
    opset_version: int = 13,
    do_constant_folding: bool = True,
    dynamic_batch_dim: bool = True,
):

    example_input = torch.randn(1, 3, 299, 299, dtype=torch.float32)

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
def add_class_dict(model_path, class_dict):
    # Add class dictionary as metadata to the ONNX model
    import onnx

    onnx_model = onnx.load(model_path)
    meta = onnx_model.metadata_props.add()
    meta.key = "class_dict"
    meta.value = json.dumps(class_dict)
    onnx.save(onnx_model, model_path)
    print("Added class_dict metadata to ONNX model.")


@unittest.skip
def verify_with_onnxruntime(onnx_path: str, input_name: str = "input", device: str = "cpu"):
    # Load ONNX model and show input shapes

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print("ONNX input name:", inp.name)
    print("ONNX input shape (None is dynamic):", inp.shape)
    print("ONNX input type:", inp.type)

    # Run with different batch sizes to verify dynamic batch works
    import numpy as np

    for bs in (1, 4):
        # Construct random input matching the static non-batch dims in the model (first input shape)
        # Replace this creation with the proper shape and dtype for your model if using something else.
        input_shape = [bs] + [int(d) if isinstance(d, int) else 299 for d in inp.shape[1:]]
        x = np.random.randn(*input_shape).astype(np.float32)
        print(f"Running ONNX model with batch size = {bs}, input shape = {x.shape}")
        outputs = sess.run(None, {inp.name: x})
        print(outputs)
        print("Got outputs:", [o.shape for o in outputs])

    props = sess.get_modelmeta().custom_metadata_map
    if "class_dict" in props:
        class_dict = json.loads(props["class_dict"])
        print("Loaded class_dict from ONNX metadata:", class_dict)
    else:
        print("No class_dict metadata found in ONNX model.")

    
@unittest.skip
def test_env_print():
    import sys, os
    print("PYTHON_EXECUTABLE:", sys.executable)
    print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
    print("PATH:", os.environ.get("PATH"))
    assert True


@unittest.skip
def main():
    #test_env_print()
    classes = load_class_list("models/sdzwa_southwest_v3_classes.csv")

    class_dict = {i: c['class'] for i, c in classes.iterrows()}
    print(class_dict)

    model = load_classifier("models/sdzwa_southwest_v3.pt", len(class_dict))

    export_model(model, "models/sdzwa_southwest_v3.onnx")

    # add class dict metadata
    add_class_dict("models/sdzwa_southwest_v3.onnx", class_dict)


    verify_with_onnxruntime("models/sdzwa_southwest_v3.onnx")

main()