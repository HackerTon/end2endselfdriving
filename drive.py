import argparse
# import socketio
# from eventlet import wsgi
# import eventlet
# from flask import Flask
# from tensorflow import keras
# import tensorflow as tf
from io import BytesIO
from PIL import Image
import base64
import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np

# local library
from kalman import KalmanObject


# sio = socketio.Server()
# app = Flask(__name__)


class tensorTrt:
    def __init__(self, path_to_saved_model, batch_size, max_gpu_mem_size_for_trt):
        self.graph = tf.Graph()
        self.session = tf.Session()

        self.trt_graph = trt.create_inference_graph(input_graph_def=None,
                                                    outputs=None,
                                                    input_saved_model_dir=path_to_saved_model,
                                                    input_saved_model_tags=['serve'],
                                                    max_workspace_size_bytes=max_gpu_mem_size_for_trt,
                                                    max_batch_size=batch_size,
                                                    precision_mode='INT8')

        self.output_node = tf.import_graph_def(self.trt_graph, return_elements=['dense_4/BiasAdd:0'])

    def __del__(self):
        self.session.close()

    def infer(self, input):
        return self.session.run(self.output_node, feed_dict={'import/conv2d_input:0': input})


# @sio.on('telemetry')
# def telemetry(sid, data):
#     if data:
#         speed = data['speed']
#
#         imgstring = data['image']
#         input: Image.Image = Image.open(BytesIO(base64.b64decode(imgstring)))
#         input = input.crop((80, 47, 280, 113))
#         input = np.array(input)
#         input = input / 255
#         input = np.array([input])
#
#         if (float(speed) < 1):
#             throttle = 1
#         elif (float(speed) < 15):
#             throttle = .5
#         else:
#             throttle = 0.15
#
#         steer = tensor_rt.infer(input)
#         steer = float(steer)
#
#         # steer = kalmer.predict_and_update(observe_value=steer)
#
#         print(steer)
#
#         send_control(steering_angle=steer, throttle=throttle)
#     else:
#         sio.emit('manual', data={}, skip_sid=True)


# def send_cs

if __name__ == '__main__':
    tr = tensorTrt(pathtosavedmodel='./savemodel', batch_size=1, max_gpu_mem_size_for_trt=1024)

    while True:
        array = np.random.rand(1, 66, 200, 3)

        print(tr.infer(array))

    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument(
    #     '--model'
    # )
    #
    # args = parser.parse_args()
    #
    # model = keras.Sequential([
    #     keras.layers.Conv2D(input_shape=(66, 200, 3), filters=24,
    #                         kernel_size=5, strides=2,
    #                         padding='valid', activation='relu'),
    #     keras.layers.Conv2D(filters=36, kernel_size=5, strides=2,
    #                         padding='valid', activation='relu'),
    #     keras.layers.Conv2D(filters=48, kernel_size=5, strides=2,
    #                         padding='valid', activation='relu'),
    #     keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
    #                         padding='valid', activation='relu'),
    #     keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
    #                         padding='valid'),
    #     keras.layers.Dropout(rate=0.5),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(units=1164, activation='relu'),
    #     keras.layers.Dense(units=100, activation='relu'),
    #     keras.layers.Dense(units=50, activation='relu'),
    #     keras.layers.Dense(units=10, activation='relu'),
    #     keras.layers.Dense(units=1, activation='linear')
    # ])
    #
    # model.load_weights(filepath='e2e_original.h5')
    #
    # graph_def = GraphDef(model=model)
    #
    # trt_graph_def = tfrt.create_inference_graph(input_graph_def=graph_def.graph_def,
    #                                             outputs=graph_def.output_name,
    #                                             max_batch_size=1,
    #                                             max_workspace_size_bytes=1 << 30,
    #                                             precision_mode='FP32',
    #                                             minimum_segment_size=2)
    #
    # print(id(trt_graph_def))
    # graph_def.graph_def = trt_graph_def
    #
    # tensor_rt = TrtEngine(graph_trt=graph_def)
    #
    # app = socketio.Middleware(sio, app)
    #
    # wsgi.server(eventlet.listen(('', 4567)), app)
