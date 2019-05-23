from mobilenetv3 import MobilenetV3
import tensorflow as tf
import tensorflow_datasets as tfds

tf.enable_eager_execution()

BATCH_SIZE = 32
USE_TPU=False
NUM_CLASSES=10
EPOCH=20
DATASET='cifar10'
INPUT_SHAPE=(224,224)

mobilenet_v3 = MobilenetV3(INPUT_SHAPE,NUM_CLASSES,'small',alpha=1.0)
cos_lr = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch, _: tf.train.cosine_decay(1e-3, epoch,EPOCH)().numpy(), 1)
logging=tf.keras.callbacks.TensorBoard(log_dir='./logs', write_images=True)
mobilenet_v3.compile(tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.sparse_categorical_crossentropy,
                     metrics=["sparse_categorical_accuracy"])
if USE_TPU:
    tpu=tf.contrib.cluster_resolver.TPUClusterResolver()
    strategy=tf.contrib.tpu.TPUDistributionStrategy(tpu)
    mobilenet_v3=tf.contrib.tpu.keras_to_tpu_model(mobilenet_v3,strategy=strategy)
dataset, info = tfds.load(name=DATASET, split=[tfds.Split.TRAIN, tfds.Split.TEST], with_info=True,as_supervised=True,try_gcs=tfds.is_dataset_on_gcs(DATASET))
train_dataset, test_dataset = dataset
train_num = info.splits['train'].num_examples
test_num = info.splits['test'].num_examples

# def zca_whitening(inputs,epsilon=1e-8):
#     sigma = np.dot(inputs, inputs.T) / inputs.shape[1]  # inputs是经过归一化处理的，所以这边就相当于计算协方差矩阵
#     U, S, V = np.linalg.svd(sigma)  # 奇异分解
#     epsilon = 0.1  # 白化的时候，防止除数为0
#     ZCAMatrix = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))), U.T)  # 计算zca白化矩阵
#     return np.dot(ZCAMatrix, inputs)

# Image whiten
# def image_whiten(image):
#     return (image-tf.math.reduce_mean(image))/ tf.math.reduce_std(image)

def preprocess_image(image):
    # image=tf.image.convert_image_dtype(image,tf.float32)
    # mean=tf.reduce_mean(image)
    # std=tf.reduce_max([tf.math.reduce_std(image),1.0 / tf.sqrt(tf.reduce_prod(tf.cast(image.shape,tf.float32)))])
    # image=(image-mean)/std
    #image=tf.py_function(lambda old_image:(old_image - np.mean(old_image)) / np.std(old_image),[image],[tf.uint8])
    #image=image_whiten(image)
    #image=tf.image.random_crop(image,[*INPUT_SHAPE*3/4,3])
    image=tf.image.random_brightness(image,0.1)
    image=tf.image.random_hue(image,0.1)
    image=tf.image.random_flip_left_right(image)
    image=tf.image.resize(image, INPUT_SHAPE)
    return image

train_dataset = train_dataset.map(lambda image,label:(preprocess_image(image),label)).shuffle(10000).batch(BATCH_SIZE).prefetch(
    tf.data.experimental.AUTOTUNE).repeat()
test_dataset = test_dataset.map(lambda image,label:(tf.image.resize(image,INPUT_SHAPE),label)).batch(BATCH_SIZE).prefetch(
    tf.data.experimental.AUTOTUNE).repeat()
mobilenet_v3.fit(train_dataset, epochs=EPOCH, steps_per_epoch=max(1,train_num//BATCH_SIZE), validation_data=test_dataset,validation_steps=max(1,test_num//BATCH_SIZE),callbacks=[cos_lr])
mobilenet_v3.save_weights('./mobilenetv3_test.h5')
