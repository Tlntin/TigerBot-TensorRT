import os
import sys
import torch
from colored import stylize, fg


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
sys.path.append(project_dir)


from trt_export.pykernel_no_past import KernelNoPast


def check_value(pre_value: torch.Tensor, true_value: torch.Tensor, diff=1e-3):
    if pre_value.shape != true_value.shape:
        raise Exception("compare shape must be same!")
    max_diff = (pre_value - true_value).abs_().max().item()
    if max_diff > diff:
        print(stylize(f"compare diff failed, diff is {max_diff}", fg("red")))
    else:
        print(stylize("compare diff OK!", fg("green")))
    return max_diff


def test():
    assert torch.cuda.is_available(), print("you must has cuda to run TensorRT")
    output_dir = os.path.join(project_dir, "output")
    model_dir = os.path.join(output_dir, "models")
    engine_path1 = os.path.join(model_dir, "TigerBot_bs1_no_past.plan")
    pt_output_dir = os.path.join(output_dir, "pt_output")
    input_path = os.path.join(pt_output_dir, "pt_input1.pt")
    output_path = os.path.join(pt_output_dir, "pt_output1.pt")
    device = torch.device("cuda:0")
    input_dict = torch.jit.load(input_path)
    batch_size = 1
    num_layers = 30
    output_dict = torch.jit.load(output_path)
    input_ids = input_dict.input_ids.int().to(device)
    attention_mask = input_dict.attention_mask.int().to(device)
    input_tensors = (input_ids, attention_mask)
    kernel = KernelNoPast(engine_path1, batch_size, num_layers)
    output_tensors = kernel.forward(input_tensors)

    # compare output
    max_diff_ = 0
    for i in range(num_layers):
        true_present_key = getattr(output_dict, f"present.{i}.key").to(device)
        true_present_value = getattr(output_dict, f"present.{i}.value").to(device)
        pre_present_key = output_tensors[i * 2]
        pre_present_value = output_tensors[i * 2 + 1]
        print("=" * 20)
        print(f"compare present.{i}.key")
        temp_diff = check_value(pre_present_key, true_present_key)
        if temp_diff > max_diff_:
            max_diff_ = temp_diff

        print("=" * 20)
        print(f"compare present.{i}.key")
        temp_diff = check_value(pre_present_value, true_present_value)
        if temp_diff > max_diff_:
            max_diff_ = temp_diff
    print(f"max diff is {max_diff_}")




if __name__ == "__main__":
    test()



