import os
import torch
import argparse
from transformers import AutoTokenizer
from transformers.models.bloom.modeling_bloom import BloomForCausalLM


def get_prompt(query, history=None):
    if not history:
        prompt = "\n\n### Instruction:\n{}\n\n### Response:\n".format(query)
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "\n\n### Instruction:\n{}\n\n### Response:\n{}".format(old_query, response)
        prompt += "\n\n### Instruction:\n{}\n\n### Response:\n".format(query)
    return prompt


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
parser = argparse.ArgumentParser(description='export pytorch model to onnx')
parser.add_argument(
    '--data_type',
    default="fp32",
    help='use fp16/fp32 to export input/output, Defualt is fp32'
)

args = parser.parse_args()
if args.data_type == "fp16":
    device = 'cuda'
else:
    device = 'cpu'

output_dir = os.path.join(project_dir, "output")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
pt_out_dir = os.path.join(output_dir, "pt_output")
if not os.path.exists(pt_out_dir):
    os.mkdir(pt_out_dir)

# save input tensor
pt_input_path1 = os.path.join(pt_out_dir, "pt_input1.pt")
pt_input_path2 = os.path.join(pt_out_dir, "pt_input2.pt")
pt_input_dict1 = dict()
pt_input_dict2 = dict()
# save output tensor
pt_output_path1 = os.path.join(pt_out_dir, "pt_output1.pt")
pt_output_path2 = os.path.join(pt_out_dir, "pt_output2.pt")
pt_output_dict1 = dict()
pt_output_dict2 = dict()


class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])


query = "你的名字？"
history = [
    (
        "你好",
        "你好！很高兴见到你。我是一个人工智能助手，我一直在等待机会来帮助您。有什么我可以为您做的吗？"
    )
]
prompt = get_prompt(query, history)

model_dir = os.path.join(project_dir, "tigerbot-7b-sft")
# get model
def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip
model = BloomForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
# get tokenizer
max_input_length: int = 512
max_generate_length: int = 1024
tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        cache_dir=None,
        model_max_length=max_generate_length,
        padding_side="left",
        truncation_side='left',
        padding=True,
        truncation=True
    )
if tokenizer.model_max_length is None or tokenizer.model_max_length > max_generate_length:
    tokenizer.model_max_length = max_generate_length
if device == "cuda":
    model = model.half().cuda()
else:
    model = model.float().cpu()
model.eval()


# to do: test chat speed

# --- prepare data for input1 ---
inputs_1 = tokenizer(
    prompt,
    return_tensors='pt',
    truncation=True, max_length=max_input_length
)
inputs_1 = {k: v.to(device) for k, v in inputs_1.items()}
input_ids_1 = inputs_1["input_ids"]
attention_mask_1 = inputs_1["attention_mask"]
# save input1
pt_input_dict1["input_ids"] = input_ids_1[:1].detach().cpu()
pt_input_dict1["attention_mask"] = attention_mask_1[:1].detach().cpu()
input_container1 = torch.jit.script(Container(pt_input_dict1))
input_container1.save(pt_input_path1)

output_dict1 = model.forward(
    input_ids=input_ids_1,
    attention_mask=attention_mask_1,
)


# this underline use to debug forward with past_key_values
"""
generation_kwargs = {
    "top_p": 0.95,
    "temperature": 0.8,
    "max_length": max_generate_length,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
    "early_stopping": True,
    "no_repeat_ngram_size": 4,
}
output = model.generate(**inputs_1, **generation_kwargs)
"""

# save output1 logists
pt_output_dict1["logits"] = output_dict1["logits"][:1].detach().cpu()
past_key_values_1 = output_dict1["past_key_values"]
print("one past_key_shape for input 1 is ", past_key_values_1[0][0].shape)
print("logits for input1 shape is ", output_dict1["logits"].shape)

# --- prepare data for input2 ---
input_ids_2 = torch.tensor([[41381]], device=device)
batch_size, seq_length = input_ids_1.shape
attention_mask_2 = torch.ones([batch_size, seq_length + 1], device=device)
# input_ids2 = torch.cat((input_ids2, input_ids2), dim=0)
# position_ids2 = torch.cat((position_ids2, position_ids2), dim=0)
# attention_mask2 = torch.cat((attention_mask2, attention_mask2), dim=0)
output_dict2 = model.forward(
    input_ids=input_ids_2,
    attention_mask=attention_mask_2,
    past_key_values=past_key_values_1,
)
past_key_values_2 = output_dict2["past_key_values"]
print("one past_key_shape for input 2 is ", past_key_values_2[0][0].shape)
print("logits for input2 shape is ", output_dict2["logits"].shape)

# save input2
pt_input_dict2["input_ids"] = input_ids_2[:1].detach().cpu()
pt_input_dict2["attention_mask"] = attention_mask_2[:1].detach().cpu()

# save logits2
pt_output_dict2["logits"] = output_dict2["logits"][:1].detach().cpu()
# save layer number
pt_output_dict1["num_layers"] = model.config.n_layer
pt_output_dict2["num_layers"] = model.config.n_layer

for layer_idx in range(model.config.n_layer):
    # --- input key and value ---
    past_key_name = f"past_key_values.{layer_idx}.key"
    past_value_name = f"past_key_values.{layer_idx}.value"
    # --- output key and value ---
    present_key_name = f"present.{layer_idx}.key"
    present_value_name = f"present.{layer_idx}.value"

    # save output1 present_key_values 
    present_key = past_key_values_1[layer_idx][0].detach().cpu()
    present_value = past_key_values_1[layer_idx][1].detach().cpu()
    pt_output_dict1[present_key_name] = present_key
    pt_output_dict1[present_value_name] = present_value

    # save input2 past_key_values
    # input2 past_key_values is same as output1 present_key_values
    pt_input_dict2[past_key_name] = present_key
    pt_input_dict2[past_value_name] = present_value

    # save output2 present_key_values
    present_key2 = past_key_values_2[layer_idx][0].detach().cpu()
    present_value2 = past_key_values_2[layer_idx][1].detach().cpu()
    pt_output_dict2[present_key_name] = present_key2
    pt_output_dict2[present_value_name] = present_value2

# save output1
output1_container = torch.jit.script(Container(pt_output_dict1))
output1_container.save(pt_output_path1)

# save input2
input2_container = torch.jit.script(Container(pt_input_dict2))
input2_container.save(pt_input_path2)

# save output2
output2_container = torch.jit.script(Container(pt_output_dict2))
output2_container.save(pt_output_path2)