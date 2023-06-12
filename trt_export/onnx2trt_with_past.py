import tensorrt as trt
import os
import time
from colored import fg, stylize
import json
# import onnx
from polygraphy.backend.trt import network_from_onnx_path
from itertools import tee
from tensorrt import MemoryPoolType, PreviewFeature

# default is 1, maybe you can try 2, 4, 8, 16
batch_size = 1
# reduce compile time
use_time_cache = True
max_length = 512
opt_length = max_length // 2
gen_max_length = 1024
gen_opt_length = gen_max_length // 2
# if use force use fp16, may reduce the accuracy and memory usage
force_use_fp16 = False
# default 3, max 5, 5 is the best but need more GPU memory and time
builder_optimization_level = 3
# lower memory GPU can try this option with True \
# it can use CPU memory/CPU compute to run some layers, but may reduce the speed
all_gpu_fallback = False

if batch_size > 1 and builder_optimization_level != 5:
    raise Exception("batch size > 1, please use builder_optimization_level = 5")


class MyLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        if severity == trt.Logger.ERROR:
            print(stylize("[ERROR] " + msg, fg("red")))  # 红色字体
        elif severity == trt.Logger.WARNING:
            print(stylize("[WARNING] " + msg, fg("yellow")))  # 黄色字体
        elif severity == trt.Logger.INTERNAL_ERROR:
            print(stylize("[INTERNAL_ERROR] " + msg, fg("red")))  # 红色字体
        elif severity == trt.Logger.INFO:
            print(stylize("[INFO] " + msg, fg("green")))  # 绿色字体
        elif severity == trt.Logger.VERBOSE:
            print(stylize("[VERBOSE] " + msg, fg("blue")))  # 蓝色字体
        else:
            print("[UNKNOWN] " + msg)


def get_network_definition(trt_network):
    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    layer_type_set = set() 
    num_layers = trt_network.num_layers 
    indices = list(range(num_layers))
    for i, i_next in pairwise(indices):
        layer = trt_network.get_layer(i)
        l_next = trt_network.get_layer(i_next)
        # print(layer.name, l_next.name, layer.type, l_next.type)
        layer_type_set.add(str(layer.type))
        if layer.type == trt.LayerType.SOFTMAX:
            layer.precision = trt.DataType.FLOAT
            if l_next is not None:
                l_next.precision = trt.DataType.FLOAT
        else:
            if force_use_fp16 and layer.get_output_type(0) == trt.float32: 
                layer.precision = trt.DataType.HALF
    layer_type_path = os.path.join(output_dir, "layer_type.json")
    with open(layer_type_path, "wt") as f:
        json.dump(list(layer_type_set), f, indent=4)
    return trt_network


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
model_dir = os.path.join(output_dir, "models")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
onnx_output_dir = os.path.join(output_dir, "onnx_output")
onnx_model_dir = os.path.join(onnx_output_dir, "tigerbot-7b-sft-fp32")
onnx_model_path = os.path.join(onnx_model_dir, "decoder_with_past_model.onnx")
time_cache_dir = os.path.join(output_dir, "time_cache")
if not os.path.exists(time_cache_dir):
    os.mkdir(time_cache_dir)
time_cache_path = os.path.join(time_cache_dir, "time_cache_with_past.cache")
tensorrt_engine_path = os.path.join(model_dir, f"TigerBot_bs{batch_size}_with_past.plan")
builder = trt.Builder(MyLogger())
config = builder.create_builder_config()
profile = builder.create_optimization_profile()
profile.set_shape(
    "input_ids",
    (1, 1),
    (batch_size, 1),
    (batch_size, 1),
)
profile.set_shape(
    "attention_mask",
    (1, 1),
    (batch_size, 1),
    (batch_size, 1),
)
for i in range(30):
    profile.set_shape(
        f"past_key_values.{i}.key",
        (1 * 32, 128, 1),
        (batch_size * 32, 128, gen_opt_length - 1),
        (batch_size * 32, 128, gen_max_length - 1),
    )
    profile.set_shape(
        f"past_key_values.{i}.value",
        (1 * 32, 1, 128),
        (batch_size * 32, gen_opt_length - 1, 128),
        (batch_size * 32, gen_max_length - 1, 128),
    )
config.add_optimization_profile(profile)
# use fp16
config.flags = config.flags | (1 << int(trt.BuilderFlag.FP16))
# disable tf32
config.flags = config.flags & ~(1 << int(trt.BuilderFlag.TF32))
# use obey precision constraints
# config.flags = config.flags | (1 << int(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS))
# config.set_memory_pool_limit(MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
print("FP16 is ON? ", config.get_flag(trt.BuilderFlag.FP16))
# use prewview features
preview_features = [
    # PreviewFeature.PROFILE_SHARING_0806,
    # PreviewFeature.FASTER_DYNAMIC_SHAPES_0805,
    # PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805
]
for feature in preview_features:
    config.set_preview_feature(feature, True)
config.builder_optimization_level = builder_optimization_level

# use time cache
time_cache = b""
# read time cache
if use_time_cache:
    if os.path.exists(time_cache_path):
        time_cache = open(time_cache_path, "rb").read()
        if time_cache is None:
            time_cache = b""
            print(stylize("read time cache failed", fg("yellow")))
            print(stylize("we will use empty cache to replace it!", fg("yellow")))
        else:
            print(stylize("read time cache from {}".format(time_cache_path), fg("green")))
    else:
        time_cache = b""
        print(stylize(
            "time cache file path: {} not exists".format(time_cache_path) +
            "we will use empty cache to replace it!",
            fg("yellow"))
        )

    # set time cache
    cache = config.create_timing_cache(time_cache)
    config.set_timing_cache(cache, True)


# load onnx model
print("loading onnx model from ", onnx_model_path)
_, network, _ = network_from_onnx_path(onnx_model_path)

# this may need more memory to compile, about > 24G
# if you don't have enough memory, you can note this line
# network = get_network_definition(network)
"""
print("=============tensorRT inference config =====================")
if builder_optimization_level == 3:
    print("==tensorRT engine begin compile, maybe you need wait 10-25 minute ==")
elif builder_optimization_level == 5:
    print("==tensorRT engine begin compile, maybe you need wait 30-60 minute ==")
else:
    print("==tensorRT engine begin compile, maybe you need wait a moment ==")

"""

# trt_engine = engine_from_network(
#     (trt_builder, network, onnx_parser),
#     trt_inference_config
# )
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is not None:
    # 保存引擎到文件
    with open(tensorrt_engine_path, "wb") as f:
        f.write(serialized_engine)
    # save_engine(trt_engine, tensorrt_engine_path)
    print("engine save in ", tensorrt_engine_path)
    print("==tensorRT engine compile done==")
else:
    raise RuntimeError("build engine failed")

# save time cache
if use_time_cache and not os.path.exists(time_cache_path):
    time_cache = config.get_timing_cache()
    time_cache_data = time_cache.serialize()
    if time_cache is not None:
        open(time_cache_path, "wb").write(time_cache_data)
        print(stylize("save time cache to {}".format(time_cache_path), fg("green")))
    else:
        print(stylize("can't found any time cache", stylize("yellow")))