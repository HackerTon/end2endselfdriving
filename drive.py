import argparse
import socketio
from eventlet import wsgi
import eventlet
from flask import Flask
from tensorflow import keras
import tensorflow as tf
from io import BytesIO
from PIL import Image
import base64
import numpy as np
from tensorflow.contrib import tensorrt as tfrt

# local library
from kalman import KalmanObject

sio = socketio.Server()
app = Flask(__name__)


class GraphDef(object):
    def __init__(self, model: tf.keras.Model):
        with tf.keras.backend.get_session() as kerasSess:
            input_tensor: tf.Tensor = tf.placeholder(dtype=tf.float32,
                                                     shape=(None, 66, 200, 3),
                                                     name='input_tensor')

            tf.keras.backend.set_learning_phase(0)

            model = model(inputs=input_tensor)

            output_name = model.name[:-2]

            model_graph = kerasSess.graph.as_graph_def()

            finalize_graph = tf.graph_util.convert_variables_to_constants(sess=kerasSess,
                                                                          input_graph_def=model_graph,
                                                                          output_node_names=[output_name])

            finalize_graph = tf.graph_util.remove_training_nodes(input_graph=finalize_graph)

            self.graph_def = finalize_graph
            self.input_name = [input_tensor.name[:-2]]
            self.output_name = [output_name]


class TrtEngine(object):
    def __init__(self, graph_trt: GraphDef):
        graph = tf.Graph()

        with graph.as_default():
            input_op, output_op = tf.import_graph_def(
                graph_def=graph_trt.graph_def,
                return_elements=graph_trt.input_name + graph_trt.output_name
            )

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.5,
            allow_growth=True
        ))

        self.trt_session = tf.Session(graph=graph, config=config)

        self.input_tensor = input_op.outputs[0]
        self.output_tensor = output_op.outputs[0]

    def infer(self, input_data):
        result = self.trt_session.run(self.output_tensor,
                                      feed_dict={self.input_tensor: input_data})

        return result


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = data['speed']

        imgstring = data['image']
        input: Image.Image = Image.open(BytesIO(base64.b64decode(imgstring)))
        input = input.crop((80, 47, 280, 113))
        input = np.array(input)
        input = input / 255
        input = np.array([input])

        if (float(speed) < 1):
            throttle = 1
        elif (float(speed) < 15):
            throttle = .5
        else:
            throttle = 0.15

        steer = tensor_rt.infer(input)
        steer = float(steer)

        # steer = kalmer.predict_and_update(observe_value=steer)

        print(steer)

        send_control(steering_angle=steer, throttle=throttle)
    else:
        sio.emit('manual', data={}, skip_sid=True)


def send_control(steering_angle, throttle):
    sio.emit(
        'steer',
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model'
    )

    args = parser.parse_args()

    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=(66, 200, 3), filters=24,
                            kernel_size=5, strides=2,
                            padding='valid', activation='relu'),
        keras.layers.Conv2D(filters=36, kernel_size=5, strides=2,
                            padding='valid', activation='relu'),
        keras.layers.Conv2D(filters=48, kernel_size=5, strides=2,
                            padding='valid', activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                            padding='valid', activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                            padding='valid'),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(units=1164, activation='relu'),
        keras.layers.Dense(units=100, activation='relu'),
        keras.layers.Dense(units=50, activation='relu'),
        keras.layers.Dense(units=10, activation='relu'),
        keras.layers.Dense(units=1, activation='linear')
    ])

    model.load_weights(filepath='e2e_original.h5')

    graph_def = GraphDef(model=model)

    trt_graph_def = tfrt.create_inference_graph(input_graph_def=graph_def.graph_def,
                                                outputs=graph_def.output_name,
                                                max_batch_size=1,
                                                max_workspace_size_bytes=1 << 30,
                                                precision_mode='FP32',
                                                minimum_segment_size=2)

    print(id(trt_graph_def))
    graph_def.graph_def = trt_graph_def

    tensor_rt = TrtEngine(graph_trt=graph_def)

    app = socketio.Middleware(sio, app)

    wsgi.server(eventlet.listen(('', 4567)), app)
