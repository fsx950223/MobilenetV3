import tensorflow as tf

class Squeeze(tf.keras.layers.Layer):
    def call(self, input):
        x = tf.squeeze(input, [1])
        x = tf.squeeze(x, [1])
        return x

class Bneck(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 expansion_filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 use_se=False,
                 activation=tf.nn.relu6,
                 **kwargs):
        super(Bneck, self).__init__(**kwargs)
        self.filters = filters
        self.expansion_filters = expansion_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.use_se = use_se
        self.activation = activation
        self.expand_conv2d = tf.keras.layers.Conv2D(self.expansion_filters, 1, padding='same', use_bias=False)
        self.expand_bn = tf.keras.layers.BatchNormalization()
        self.zero_padding2d = tf.keras.layers.ZeroPadding2D(((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2))
        self.depthwise_conv2d = tf.keras.layers.DepthwiseConv2D(self.kernel_size, strides=self.strides, use_bias=False,
                                                                padding='same' if self.strides == 1 else 'valid')
        self.depthwise_bn = tf.keras.layers.BatchNormalization()
        self.se = SeBlock()
        self.project_conv2d = tf.keras.layers.Conv2D(self.filters, kernel_size=1, padding='same', use_bias=False)
        self.project_bn = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.built = True

    def call(self, inputs):
        x = self.expand_conv2d(inputs)
        x = self.expand_bn(x)
        x = self.activation(x)
        if self.strides == 2:
            x = self.zero_padding2d(x)
        x = self.depthwise_conv2d(x)
        x = self.depthwise_bn(x)
        x = self.activation(x)
        x = self.project_conv2d(x)
        x = self.project_bn(x)
        if self.in_channels == self.filters and self.strides == 1:
            if self.use_se:
                x = self.se(x)
            x = self.add([inputs, x])
        return x


class SeBlock(tf.keras.layers.Layer):
    def __init__(self, reduction=4,**kwargs):
        super(SeBlock,self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        self.average_pool = tf.keras.layers.AveragePooling2D((int(input_shape[1]),int(input_shape[2])))
        self.conv1 = tf.keras.layers.Conv2D(int(input_shape[-1]) // self.reduction, 1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(int(input_shape[-1]), 1, use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.h_swish=HSwish()
        self.built = True

    def call(self, inputs):
        x = self.average_pool(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu6(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.h_swish(x)
        return x

def h_swish(inputs):
    return inputs * tf.nn.relu6(inputs + 3) / 6

class HSwish(tf.keras.layers.Layer):
  def call(self, inputs):
    return h_swish(inputs)

def MobilenetV3(input_shape,num_classes, size="large", include_top=True):
    input = tf.keras.layers.Input([*input_shape, 3])
    if size not in ['large', 'small']:
        raise ValueError('size should be large or small')
    if size == "large":
        x = tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = HSwish()(x)
        x = Bneck(16, 16, 3, strides=1, padding='same', use_se=False, activation=tf.nn.relu6)(x)
        x = Bneck(24, 64, 3, strides=2, padding='valid', use_se=False, activation=tf.nn.relu6)(x)
        x = Bneck(24, 72, 3, strides=1, padding='same', use_se=False, activation=tf.nn.relu6)(x)
        x = Bneck(40, 72, 5, strides=2, padding='valid', use_se=True, activation=tf.nn.relu6)(x)
        x = Bneck(40, 120, 5, strides=1, padding='same', use_se=True, activation=tf.nn.relu6)(x)
        x = Bneck(40, 120, 5, strides=1, padding='same', use_se=True, activation=tf.nn.relu6)(x)
        x = Bneck(80, 240, 3, strides=2, padding='valid', use_se=False, activation=h_swish)(x)
        x = Bneck(80, 200, 3, strides=1, padding='same', use_se=False, activation=h_swish)(x)
        x = Bneck(80, 184, 3, strides=1, padding='same', use_se=False, activation=h_swish)(x)
        x = Bneck(80, 184, 3, strides=1, padding='same', use_se=False, activation=h_swish)(x)
        x = Bneck(112, 480, 3, strides=1, padding='same', use_se=True, activation=h_swish)(x)
        x = Bneck(112, 672, 3, strides=1, padding='same', use_se=True, activation=h_swish)(x)
        x = Bneck(160, 672, 5, strides=2, padding='valid', use_se=True, activation=h_swish)(x)
        x = Bneck(160, 960, 5, strides=1, padding='same', use_se=True, activation=h_swish)(x)
        x = Bneck(160, 960, 5, strides=1, padding='same', use_se=True, activation=h_swish)(x)
        x = tf.keras.layers.Conv2D(960, 1, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output = HSwish()(x)
    else:
        x = tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = HSwish()(x)
        x = Bneck(16, 16, 3, strides=2, padding='valid', use_se=True, activation=tf.nn.relu6)(x)
        x = Bneck(24, 72, 3, strides=2, padding='valid', use_se=False, activation=tf.nn.relu6)(x)
        x = Bneck(24, 88, 3, strides=1, padding='same', use_se=False, activation=tf.nn.relu6)(x)
        x = Bneck(40, 96, 5, strides=2, padding='valid', use_se=True, activation=h_swish)(x)
        x = Bneck(40, 240, 5, strides=1, padding='same', use_se=True, activation=h_swish)(x)
        x = Bneck(40, 240, 5, strides=1, padding='same', use_se=True, activation=h_swish)(x)
        x = Bneck(48, 120, 5, strides=1, padding='same', use_se=True, activation=h_swish)(x)
        x = Bneck(48, 144, 5, strides=1, padding='same', use_se=True, activation=h_swish)(x)
        x = Bneck(96, 288, 5, strides=2, padding='valid', use_se=True, activation=h_swish)(x)
        x = Bneck(96, 576, 5, strides=1, padding='same', use_se=True, activation=h_swish)(x)
        x = Bneck(96, 576, 5, strides=1, padding='same', use_se=True, activation=h_swish)(x)
        x = tf.keras.layers.Conv2D(576, 1, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = HSwish()(x)
        output=tf.keras.layers.Add()([x, SeBlock()(x)])
    if include_top:
        output = tf.keras.layers.AveragePooling2D(pool_size=x.shape[1:3])(output)
        output = tf.keras.layers.Conv2D(1280, 1)(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = HSwish()(output)
        output = tf.keras.layers.Conv2D(num_classes, 1)(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Softmax()(output)
        output = Squeeze()(output)
    return tf.keras.Model(input,output)