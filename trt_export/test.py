import tensorrt as trt
import ctypes
import os
import logging
import argparse
import sys

#TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
#TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger()
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

"""Takes an ONNX file and creates a TensorRT engine to run inference with"""
max_batch_size = 2048
save_engine = True

def mark_outputs(network):
    # Mark last layer's outputs if not already marked
    # NOTE: This may not be correct in all cases
    last_layer = network.get_layer(network.num_layers-1)
    if not last_layer.num_outputs:
        logger.error("Last layer contains no outputs.")
        return

    for i in range(last_layer.num_outputs):
        network.mark_output(last_layer.get_output(i))

def check_network(network):
    if not network.num_outputs:
        logger.warning("No output nodes found, marking last layer's outputs as network outputs. Correct this if wrong.")
        mark_outputs(network)

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    max_len = max([len(inp.name) for inp in inputs] + [len(out.name) for out in outputs])

    logger.debug("=== Network Description ===")
    for i, inp in enumerate(inputs):
        logger.debug("Input  {0} | Name: {1:{2}} | Shape: {3}".format(i, inp.name, max_len, inp.shape))
    for i, out in enumerate(outputs):
        logger.debug("Output {0} | Name: {1:{2}} | Shape: {3}".format(i, out.name, max_len, out.shape))


def create_optimization_profiles(builder, inputs, batch_sizes=[1, 800, 1600]):
    # Check if all inputs are fixed explicit batch to create a single profile and avoid duplicates
    profiles = builder.create_optimization_profile()
    import pdb
    #pdb.set_trace()
    for inp in inputs:
        shape = inp.shape[1:]
        # Check if fixed explicit batch
        if inp.shape[0] > -1:
            bs = inp.shape[0]
            profiles.set_shape(inp.name, min=(bs, *shape), opt=(bs, *shape), max=(bs, *shape))
        else:
            #profiles.set_shape(inp.name, min=(batch_sizes[0], *shape), opt=(batch_sizes[1], *shape), max=(batch_sizes[2], *shape))
            profiles.set_shape(inp.name, min=(batch_sizes[0], 1), opt=(batch_sizes[1], 128), max=(batch_sizes[2], 256))
    return profiles

def add_profiles(config, inputs, profile):
    logger.debug("=== Optimization Profiles ===")
    for inp in inputs:
        _min, _opt, _max = profile.get_shape(inp.name)
        logger.debug("{} - Min {} Opt {} Max {}".format(inp.name, _min, _opt, _max))
    config.add_optimization_profile(profile)
    return

def main():
    parser = argparse.ArgumentParser(description="Creates a TensorRT engine from the provided ONNX file.\n")
    parser.add_argument("--onnx", required=True, help="The ONNX model file to convert to TensorRT")
    parser.add_argument("-o", "--output", type=str, default="model.engine", help="The path at which to write the engine")
    parser.add_argument("-b", "--max-batch-size", type=int, default=2048, help="The max batch size for the TensorRT engine input")
    parser.add_argument("-v", "--verbosity", action="count", help="Verbosity for logging. (None) for ERROR, (-v) for INFO/WARNING/ERROR, (-vv) for VERBOSE.")
    parser.add_argument("--explicit-batch", action='store_true', help="Set trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH.")
    parser.add_argument("--explicit-precision", action='store_true', help="Set trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION.")
    parser.add_argument("--gpu-fallback", action='store_true', help="Set trt.BuilderFlag.GPU_FALLBACK.")
    parser.add_argument("--refittable", action='store_true', help="Set trt.BuilderFlag.REFIT.")
    parser.add_argument("--debug", action='store_true', help="Set trt.BuilderFlag.DEBUG.")
    parser.add_argument("--strict-types", action='store_true', help="Set trt.BuilderFlag.STRICT_TYPES.")
    parser.add_argument("--fp16", action="store_true", help="Attempt to use FP16 kernels when possible.")
    parser.add_argument("--int8", action="store_true", help="Attempt to use INT8 kernels when possible. This should generally be used in addition to the --fp16 flag. \
                                                             ONLY SUPPORTS RESNET-LIKE MODELS SUCH AS RESNET50/VGG16/INCEPTION/etc.")
    parser.add_argument("-p", "--preprocess_func", type=str, default=None, help="(INT8 ONLY) Function defined in 'processing.py' to use for pre-processing calibration data.")
    args, _ = parser.parse_known_args()

    # Network flags
    network_flags = 0
    args.explicit_batch = True
    if args.explicit_batch:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if args.explicit_precision:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)

    builder_flag_map = {
            'gpu_fallback': trt.BuilderFlag.GPU_FALLBACK,
            'refittable': trt.BuilderFlag.REFIT,
            'debug': trt.BuilderFlag.DEBUG,
            'strict_types': trt.BuilderFlag.STRICT_TYPES,
            'fp16': trt.BuilderFlag.FP16,
            'int8': trt.BuilderFlag.INT8,
    }

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(network_flags) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        #config.max_workspace_size = (1 << 32)  # Your workspace size

                # Set Builder Config Flags
        for flag in builder_flag_map:
            if getattr(args, flag):
                logger.info("Setting {}".format(builder_flag_map[flag]))
                config.set_flag(builder_flag_map[flag])
                # Precision flags

        if args.fp16 and not builder.platform_has_fast_fp16:
            logger.warning("FP16 not supported on this platform.")

        import pdb
        #pdb.set_trace()
        #builder.fp16_mode = True  # Default: False
        #builder.int8_mode = False  # Default: False
        #builder.strict_type_constraints = True
        #config.set_tactic_sources(1 << int(trt.TacticSource.CUDNN))
        # Parse model file
        if not os.path.exists(args.onnx):
            quit('ONNX file {} not found'.format(args.onnx))

        print('Loading ONNX file from path {}...'.format(args.onnx))
        with open(args.onnx, 'rb') as f:
            print('Beginning ONNX file parsing')
            if not parser.parse(f.read()):
                print('ERROR: Failed to parse the ONNX file: {}'.format(args.onnx))
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)

        #profile = builder.create_optimization_profile()

        batch_sizes = [1, 32, 64]
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        opt_profiles = create_optimization_profiles(builder, inputs, batch_sizes)
        add_profiles(config, inputs, opt_profiles)

        #config = builder.create_builder_config()
        #config.set_flag(trt.BuilderFlag.FP16)

        check_network(network)

        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(args.output))

        engine = builder.build_engine(network, config)
        print("Completed creating Engine")

        with open(args.output, "wb") as f:
            logger.info("Serializing engine to file: {:}".format(args.output))
            f.write(engine.serialize())
    if os.path.exists(args.output):
        print("Engine had saved to file {}".format(args.output))


if __name__ == "__main__":
    main()