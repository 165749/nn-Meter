from nn_meter import load_latency_predictor
from nn_meter.predictor import list_latency_predictors
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from nn_meter.ir_converter.frozenpb_converter import FrozenPbConverter


def import_from_keras(name, path, image_size=224):
    from models.model_provider import get_model
    if name in {
        'mobilenetv3_small_w7d20',
        'mobilenetv3_small_wd2',
        'mobilenetv3_small_w3d4',
        'mobilenetv3_small_w1',
        'mobilenetv3_small_w5d4',
        'mobilenetv3_large_w7d20',
        'mobilenetv3_large_wd2',
        'mobilenetv3_large_w3d4',
        # 'mobilenetv3_large_w1',
        'mobilenetv3_large_w5d4',
        'sepreresnet14',
        'darknet19',
        'resnext26_16x4d',
        'dpn68b',
        'bam_resnet18',
        'cbam_resnet18',
        'ghostnet',

        # With group convolutions:
        'regnetx002',
        'regnetx004',
        'regnetx006',
        'regnetx008',
        'regnetx016',
        'regnety002',
        'regnety004',
        'regnety006',
        'regnety008',
        'regnety016',
        'resnext14_16x4d',
        'resnext14_32x2d',
        'resnext14_32x4d',
        'resnext26_16x4d',
        'resnext26_32x2d',
        'resnext26_32x4d',
        'dla46xc',
        'dla60xc',
        'dpn68',
        'resnestabc14',
        'resnesta18',
    }:
        model = get_model(name, classes=1000, pretrained=False)
    else:
        model = get_model(name, classes=1000, pretrained=True)

    x = model.compute_output_shape((1, image_size, image_size, 3))
    # print(name)
    # print(x)
    # assert x == (1, 1000)
    model.summary()
    model.build((1, 224, 224, 3))
    # model.compile(optimizer='adam',
    #               loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=[tf.metrics.SparseCategoricalAccuracy()])

    full_model = tf.function(model)
    full_model = full_model.get_concrete_function(tf.TensorSpec([1, 224, 224, 3], dtype=tf.float32))
    frozen_func = convert_variables_to_constants_v2(full_model)
    graph_def = frozen_func.graph.as_graph_def()
    return graph_def

    # tf.io.write_graph(graph_or_graph_def=graph_def,
    #                   logdir=path,
    #                   name="frozen_graph.pb",
    #                   as_text=False)
    # return model
    # converter = FrozenPbConverter(graph_def)
    # graph = converter.get_flatten_graph()
    # return graph


if __name__ == '__main__':
    models = [
        'bagnet9', # New
        'bninception',
        'darknet_tiny',
        'darknet19',
        'densenet121',
        'densenet169',
        'diracnet18v2',
        'dla46c', # New
        'efficientnet_edge_small_b',
        'efficientnet_edge_medium_b',
        'efficientnet_edge_large_b',
        'fbnet_cb',
        'fdmobilenet_w1',
        'fdmobilenet_w3d4',
        'fdmobilenet_wd2',
        'fdmobilenet_wd4',
        'ghostnet',
        'hardnet39ds',
        'hardnet68ds',
        'hrnet_w18_small_v1',
        'hrnet_w18_small_v2',
        'mnasnet_b1',
        'mnasnet_a1',
        'mobilenet_w1',
        'mobilenet_w3d4',
        'mobilenet_wd2',
        'mobilenet_wd4',
        'mobilenetb_w1',
        'mobilenetb_w3d4',
        'mobilenetb_wd2',
        'mobilenetb_wd4',
        'mobilenetv2_w1',
        'mobilenetv2_w3d4',
        'mobilenetv2_wd2',
        'mobilenetv2_wd4',
        'mobilenetv2b_w1',
        'mobilenetv2b_w3d4',
        'mobilenetv2b_wd2',
        'mobilenetv2b_wd4',
        'mobilenetv3_small_w7d20',
        'mobilenetv3_small_wd2',
        'mobilenetv3_small_w3d4',
        'mobilenetv3_small_w1',
        'mobilenetv3_small_w5d4',
        'mobilenetv3_large_w7d20',
        'mobilenetv3_large_wd2',
        'mobilenetv3_large_w3d4',
        'mobilenetv3_large_w1',
        'mobilenetv3_large_w5d4',
        'peleenet',
        'preresnet10',
        'preresnet12',
        'preresnet14',
        'preresnetbc14b',
        'preresnet16',
        'preresnet18_wd4',
        'preresnet18_wd2',
        'preresnet18_w3d4',
        'preresnet18',
        'preresnetbc26b',
        'proxylessnas_cpu',
        'proxylessnas_gpu',
        'proxylessnas_mobile',
        'proxylessnas_mobile14',
        'regnetx002',
        'regnetx004',
        'regnetx006',
        'regnetx008',
        'regnetx016',
        'regnety002',
        'regnety004',
        'regnety006',
        'regnety008',
        'regnety016',
        'resnet10',
        'resnet12',
        'resnet14',
        'resnetbc14b',
        'resnet16',
        'resnet18_wd4',
        'resnet18_wd2',
        'resnet18_w3d4',
        'resnet18',
        'resnetbc26b',
        'resnext14_16x4d',
        # 'resnext14_32x2d',
        'resnext14_32x4d',
        'resnext26_16x4d',
        # 'resnext26_32x2d',
        'resnext26_32x4d',
        'sepreresnet10',
        'sepreresnet12',
        'sepreresnet14',
        'sepreresnet16',
        'sepreresnet18',
        'seresnet10',
        'seresnet12',
        'seresnet14',
        'seresnet16',
        'seresnet18',
        'spnasnet',
        'squeezenet_v1_0',
        'squeezenet_v1_1',
        'squeezeresnet_v1_0',
        'squeezeresnet_v1_1',
        'vovnet27s',
    ]

    # for predictor in list_latency_predictors():
    #     print(predictor)

    hardware_name = 'cortexA76cpu_tflite21'
    # hardware_name = 'adreno640gpu_tflite21'
    predictor = load_latency_predictor(hardware_name)

    with open("common.txt", "w") as f:
        for model in models:
            graph = import_from_keras(model, f'model_output/{model}')
            latency = predictor.predict(graph, model_type="pb")
            f.write(f"{model},{latency}\n")
