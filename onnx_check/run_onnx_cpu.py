import onnxruntime as ort
import torch
import numpy as np
import os
from colored import fg, stylize

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
onnx_output_dir = os.path.join(output_dir, "onnx_output")
onnx_model_dir = os.path.join(onnx_output_dir, "tigerbot-7b-sft-fp32")
onnx_model_path1 = os.path.join(onnx_model_dir, "decoder_model.onnx")
onnx_model_path2 = os.path.join(onnx_model_dir, "decoder_with_past_model.onnx")
new_onnx_dir = os.path.join(project_dir, "output", "new_onnx_output_fp32")
if not os.path.exists(new_onnx_dir):
    os.mkdir(new_onnx_dir)

new_onnx_path1 = os.path.join(new_onnx_dir, "decoder_model.onnx")
new_onnx_path2 = os.path.join(new_onnx_dir, "decoder_with_past_model.onnx")


def compare_value(pre_numpy: np.array, true_numpy: np.array):
    assert pre_numpy.shape == true_numpy.shape
    diff = np.abs(pre_numpy - true_numpy).max()
    if diff > 5e-4:
        print(stylize(f"diff: {diff} is_pass: failed", fg("red")))
    else:
        print(stylize(f"diff: {diff} is_pass: OK", fg("green")))
    return diff


def run_cpu_onnx_inference(
        onnx_path: str,
        new_onnx_path: str,
        input_path: str,
        output_path,
        has_past_key: bool
):
    """
    """
    # ndWtokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
    providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.optimized_model_filepath = new_onnx_path
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        onnx_path, sess_options=sess_options, providers=providers
    )
    print("provider: ", session.get_providers())
    input_dict = torch.jit.load(input_path)
    output_dict = torch.jit.load(output_path)
    input_ids = input_dict.input_ids.data.cpu().numpy().astype(np.int64)
    attention_mask = input_dict.attention_mask.data.cpu().numpy().astype(np.int64)
    logits = output_dict.logits.data.cpu().numpy()
    one_present_key = getattr(output_dict, "present.0.key").data.cpu().numpy()
    num_layers = getattr(output_dict, "num_layers")
    io_binding = session.io_binding()
    print("input number", len(session.get_inputs()))
    print("output number",len(session.get_outputs()))
    input_names = [_.name for _ in session.get_inputs()]
    print("=================input names=================")
    print(input_names)
    output_names = [_.name for _ in session.get_outputs()]
    print("=================output names=================")
    print(output_names)
    io_binding.bind_cpu_input(
        "input_ids",
        input_ids
    )
    io_binding.bind_cpu_input(
        "attention_mask",
        attention_mask
    )
    io_binding.bind_output(
        name="logits",
        device_type="cpu",
        device_id=0,
        element_type=logits.dtype,
        shape=logits.shape,
    )
    if not has_past_key:
        num_layers = 0
    for layer_idx in range(num_layers):
        input_names = [
            f"past_key_values.{layer_idx}.key",
            f"past_key_values.{layer_idx}.value"
        ]
        for name in input_names:
            past_key_values = getattr(input_dict, name).data.cpu().numpy()
            io_binding.bind_cpu_input(
                name,
                past_key_values
            )
        output_name = [
            f"present.{layer_idx}.key",
            f"present.{layer_idx}.value"
        ]
        for name in output_name:
            io_binding.bind_output(
                name,
                device_type="cpu",
                device_id=0,
                element_type=one_present_key.dtype,
                shape=one_present_key.shape,
            )


    # print(inputs)
    session.run_with_iobinding(io_binding)
    max_diff = 0
    # compare logists
    print('=' * 20)
    print("compare logits")
    pred_outputs = io_binding.copy_outputs_to_cpu()
    diff1 = compare_value(pred_outputs[0], logits)
    if diff1 > max_diff:
        max_diff = diff1

    # compare present_key_values
    for i in range(num_layers):
        key_name = f"present.{i}.key"
        value_name = f"present.{i}.value"
        print('=' * 20)
        print(f"compare {key_name}")
        # key_numpy = [key_name]
        key_true = getattr(output_dict, key_name).data.cpu().numpy()
        key_pred = pred_outputs[1 + i * 2]
        diff2 = compare_value(key_pred, key_true)
        if diff2 > max_diff:
            max_diff = diff2
        print('=' * 20)
        print(f"compare {value_name}")
        value_pred = pred_outputs[2 + i * 2]
        value_true = getattr(output_dict, value_name).data.cpu().numpy()
        diff3 = compare_value(value_pred, value_true)
        if diff3 > max_diff:
            max_diff = diff3
    print('=' * 20)
    print(f"max diff: {max_diff}")


if __name__ == "__main__":
    print("=" * 20)
    print("check onnx for forward with no past_key_value")
    input_path1 = os.path.join(output_dir, "pt_output", "pt_input1.pt")
    output_path1 = os.path.join(output_dir, "pt_output", "pt_output1.pt")
    run_cpu_onnx_inference(
        onnx_model_path1,
        new_onnx_path1,
        input_path1,
        output_path1,
        has_past_key=False
    )
    print("\n")
    print("=" * 20)
    print("check onnx for forward with past_key_value")
    input_path2 = os.path.join(output_dir, "pt_output",  "pt_input2.pt")
    output_path2 = os.path.join(output_dir, "pt_output", "pt_output2.pt")
    run_cpu_onnx_inference(
        onnx_model_path2,
        new_onnx_path2,
        input_path2,
        output_path2,
        has_past_key=True
    )