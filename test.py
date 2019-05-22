from mobilenetv3 import MobilenetV3
import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 32
NUM_CLASSES=10
DATASET='cifar10'

dataset, info = tfds.load(name=DATASET, split=tfds.Split.TEST, with_info=True,as_supervised=True,try_gcs=tfds.is_dataset_on_gcs(DATASET))
dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
mobilenet_v3 = MobilenetV3((32,32),NUM_CLASSES,'small')
mobilenet_v3.load_weights('./mobilenetv3_test.h5')
mobilenet_v3.evaluate(dataset)
