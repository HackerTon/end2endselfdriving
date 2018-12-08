# Import Library
import tensorflow as tf
import numpy as np


# from tensorflow.contrib import slim


def _function_parse(example):
    feature = {'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.float32)}

    parsed_features = tf.parse_single_example(example, feature)

    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    image.set_shape([153600])

    image = tf.cast(tf.reshape(image, [160, 320, 3]), tf.float32)

    image = tf.transpose(image, perm=[2, 0, 1])

    label = tf.cast(parsed_features['label'], tf.float32)

    return {'image': image}, label


def _function_input(filename, batch_size):
    dataset = tf.data.TFRecordDataset(filenames=filename)

    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=_function_parse, batch_size=batch_size))

    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size, count=1))

    # dataset = dataset.prefetch(buffer_size=batch_size)

    return dataset


def input_fn2():
    dataset = tf.data.Dataset.from_tensor_slices(({'x': tf.random_uniform([1, 1])}, [0.05]))

    return dataset


def cnn_baseline_model_fn(features, labels, mode, params):
    # Should be in (3, 160, 320) tensor
    input_layer = features['image']

    conv_1 = tf.layers.conv2d(inputs=input_layer, filters=24,
                              kernel_size=[5, 5], strides=(2, 2),
                              padding='same', activation=tf.nn.relu,
                              data_format='channels_first')

    conv_2 = tf.layers.conv2d(inputs=conv_1, filters=36,
                              kernel_size=[5, 5], strides=(2, 2),
                              padding='same', activation=tf.nn.relu,
                              data_format='channels_first')

    conv_3 = tf.layers.conv2d(inputs=conv_2, filters=48,
                              kernel_size=[5, 5], strides=(2, 2),
                              padding='same', activation=tf.nn.relu,
                              data_format='channels_first')

    conv_4 = tf.layers.conv2d(inputs=conv_3, filters=64,
                              kernel_size=[3, 3], strides=(1, 1),
                              padding='same', activation=tf.nn.relu,
                              data_format='channels_first')

    conv_5 = tf.layers.conv2d(inputs=conv_4, filters=64,
                              kernel_size=[3, 3], strides=(1, 1),
                              padding='same')

    conv_5 = tf.layers.flatten(conv_5)

    dense_1 = tf.layers.dense(inputs=conv_5, units=100, activation=tf.nn.relu)

    dense_2 = tf.layers.dense(inputs=dense_1, units=50, activation=tf.nn.relu)

    dense_3 = tf.layers.dense(inputs=dense_2, units=10, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense_3, training=tf.estimator.ModeKeys.PREDICT())

    output = tf.layers.dense(inputs=dropout, units=1)

    predictions = {
        "output_value": tf.reshape(output, [-1])
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions["output_value"])

    loss_log = tf.summary.scalar('loss', tensor=loss)
    predicted = tf.summary.histogram('target', predictions["output_value"])
    image = tf.summary.image('image', tensor=tf.transpose(input_layer, perm=[0, 2, 3, 1]))

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metrics_ops = {
        'error': tf.metrics.mean_squared_error(labels=labels, predictions=predictions["output_value"])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)


def main(_):
    regressor = tf.estimator.Estimator(
        model_fn=cnn_baseline_model_fn, model_dir="regressor_1",
        params={'learning_rate': 0.001}
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'image': np.random.rand(1, 3, 160, 320).astype(np.float32)},
        batch_size=1,
        shuffle=False
    )

    # regressor.train(input_fn=lambda: _function_input("/home/hackerton/data_driving/train.trf", 10), max_steps=1000)
    # print(regressor.evaluate(input_fn=lambda : _function_input("/home/hackerton/data_driving/train.trf", 20)))
    output_prediction = regressor.predict(input_fn=lambda: train_input_fn())

    print(list(output_prediction))


# TODO Write a saved_model function
# TODO Write a inference using tensor_rt
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run()
