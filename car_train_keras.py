import tensorflow as tf
import tensorflow.contrib as contrib
from tensorflow.python import keras
import os


def _read_image(image, value):
    imgString = tf.read_file(image)

    image = tf.image.decode_and_crop_jpeg(imgString, crop_window=[47, 80, 66, 200], channels=0)

    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.cast(image, tf.float32)

    image = image - tf.reduce_mean(input_tensor=image)

    return image, value


def _function_input2(filename, batch_size):
    dataset: tf.data.Dataset = tf.data.experimental.CsvDataset(filenames=filename,
                                                               record_defaults=[tf.string, tf.float32],
                                                               header=True)

    dataset = dataset.apply(tf.data.experimental.map_and_batch(_read_image,
                                                               batch_size=batch_size,
                                                               num_parallel_batches=1))

    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size))

    return dataset


def _resnet():
    model: tf.keras.Model = tf.keras.applications.ResNet50(include_top=False,
                                                           weights='imagenet',
                                                           pooling='avg',
                                                           input_shape=(1, 22, 33, 3))

    for layer in model.layers:
        layer.trainable = False

    output_resnet = model.output

    output_model = keras.layers.Dense(
        units=1, activation='linear')(output_resnet)

    trainingModel = tf.keras.Model(inputs=model.input, outputs=output_model)

    trainingModel.summary()

    earlystop = keras.callbacks.EarlyStopping(
        monitor='mean_absolute_error', min_delta=0.0001, patience=5, verbose=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir='logs/')

    trainingModel.compile(loss='mse', optimizer='nadam',
                          metrics=[keras.losses.mean_absolute_error])

    trainingModel.fit(_function_input2('driving.csv', batch_size=5),
                      epochs=30,
                      steps_per_epoch=1609,
                      callbacks=[earlystop, tensorboard])

    model.save('e2e.h5')


def _normal():
    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=(66, 200, 3), filters=24,
                            kernel_size=5, strides=2,
                            padding='valid', activation='relu'),
        keras.layers.Conv2D(filters=36, kernel_size=5, strides=2,
                            padding='valid', activation='relu'),
        keras.layers.Conv2D(filters=48, kernel_size=5, strides=2,
                            padding='valid', activation='relu'),
        keras.layers.BatchNormalization(fused=True),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                            padding='valid', activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                            padding='valid'),
        keras.layers.BatchNormalization(fused=True),
        keras.layers.Flatten(),
        keras.layers.Dense(units=1164, activation='relu'),
        keras.layers.Dense(units=100, activation='relu'),
        keras.layers.Dense(units=50, activation='relu'),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(units=10, activation='relu'),
        keras.layers.BatchNormalization(fused=True),
        keras.layers.Dense(units=1, activation='linear')
    ])

    for node in model.outputs:
        print(node)

    model.summary()

    savedmodel = contrib.saved_model.save_keras_model(model=model, saved_model_path='./savemodel')

    # if os.path.isfile('e2e.h5'):
    #     model.load_weights('e2e.h5')
    #
    # model.compile(loss='mse',
    #               optimizer=tf.keras.optimizers.Nadam(lr=0.0001, schedule_decay=0.0001),
    #               metrics=[keras.losses.mean_absolute_error])
    #
    # model.summary()
    #
    # checkpoint = keras.callbacks.ModelCheckpoint('checkpoints/weights.hdf5',
    #                                              monitor='val_loss',
    #                                              save_best_only=True)
    #
    # earlystop = keras.callbacks.EarlyStopping(monitor='mean_absolute_error',
    #                                           min_delta=0.001,
    #                                           patience=5,
    #                                           verbose=1)
    #
    # tensorboard = keras.callbacks.TensorBoard(log_dir='logs/')
    #
    # # N total = 8045
    # model.fit(_function_input2('driving.csv', batch_size=2 * 5 * 11),
    #           epochs=1000, steps_per_epoch=73, verbose=1, callbacks=[tensorboard, checkpoint])
    #
    # model.save_weights('e2e.h5', overwrite=True)


def main():
    _normal()


if __name__ == '__main__':
    main()
