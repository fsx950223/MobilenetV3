# MobilenetV3
## Reference
[Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
## Requirement
tensorflow-datasets             1.0.2 
tensorflow                      1.13.1
## Usage
### Train
``` python
python train.py
```
### Test
``` python
python test.py
```
### Performance
MobilenetV3 small 32x32:
```
78/78 [==============================] - 4s 52ms/step - loss: 2.3378 - sparse_categorical_accuracy: 0.1002
390/390 [==============================] - 49s 126ms/step - loss: 1.8914 - sparse_categorical_accuracy: 0.3221 - val_loss: 2.3378 - val_sparse_categorical_accuracy: 0.1002

Epoch 00002: LearningRateScheduler reducing learning rate to 0.0009938442.
Epoch 2/20
78/78 [==============================] - 3s 39ms/step - loss: 1.4659 - sparse_categorical_accuracy: 0.4778
390/390 [==============================] - 37s 95ms/step - loss: 1.5207 - sparse_categorical_accuracy: 0.4571 - val_loss: 1.4659 - val_sparse_categorical_accuracy: 0.4778

Epoch 00003: LearningRateScheduler reducing learning rate to 0.00097552827.
Epoch 3/20
78/78 [==============================] - 3s 39ms/step - loss: 1.5232 - sparse_categorical_accuracy: 0.4734
390/390 [==============================] - 32s 83ms/step - loss: 1.4056 - sparse_categorical_accuracy: 0.5052 - val_loss: 1.5232 - val_sparse_categorical_accuracy: 0.4734

Epoch 00004: LearningRateScheduler reducing learning rate to 0.0009455033.
Epoch 4/20
78/78 [==============================] - 3s 39ms/step - loss: 1.3620 - sparse_categorical_accuracy: 0.5288
390/390 [==============================] - 34s 87ms/step - loss: 1.3271 - sparse_categorical_accuracy: 0.5362 - val_loss: 1.3620 - val_sparse_categorical_accuracy: 0.5288

Epoch 00005: LearningRateScheduler reducing learning rate to 0.0009045085.
Epoch 5/20
78/78 [==============================] - 3s 39ms/step - loss: 1.2827 - sparse_categorical_accuracy: 0.5558
390/390 [==============================] - 32s 82ms/step - loss: 1.2645 - sparse_categorical_accuracy: 0.5608 - val_loss: 1.2827 - val_sparse_categorical_accuracy: 0.5558

Epoch 00006: LearningRateScheduler reducing learning rate to 0.0008535535.
Epoch 6/20
78/78 [==============================] - 4s 45ms/step - loss: 1.4777 - sparse_categorical_accuracy: 0.4679
390/390 [==============================] - 34s 86ms/step - loss: 1.2359 - sparse_categorical_accuracy: 0.5711 - val_loss: 1.4777 - val_sparse_categorical_accuracy: 0.4679

Epoch 00007: LearningRateScheduler reducing learning rate to 0.00079389266.
Epoch 7/20
78/78 [==============================] - 3s 39ms/step - loss: 1.2299 - sparse_categorical_accuracy: 0.5765
390/390 [==============================] - 32s 82ms/step - loss: 1.2234 - sparse_categorical_accuracy: 0.5768 - val_loss: 1.2299 - val_sparse_categorical_accuracy: 0.5765

Epoch 00008: LearningRateScheduler reducing learning rate to 0.00072699535.
Epoch 8/20
78/78 [==============================] - 3s 39ms/step - loss: 1.1509 - sparse_categorical_accuracy: 0.6074
390/390 [==============================] - 32s 82ms/step - loss: 1.1740 - sparse_categorical_accuracy: 0.5935 - val_loss: 1.1509 - val_sparse_categorical_accuracy: 0.6074

Epoch 00009: LearningRateScheduler reducing learning rate to 0.0006545085.
Epoch 9/20
78/78 [==============================] - 3s 39ms/step - loss: 1.1879 - sparse_categorical_accuracy: 0.5973
390/390 [==============================] - 34s 88ms/step - loss: 1.1341 - sparse_categorical_accuracy: 0.6109 - val_loss: 1.1879 - val_sparse_categorical_accuracy: 0.5973

Epoch 00010: LearningRateScheduler reducing learning rate to 0.00057821727.
Epoch 10/20
78/78 [==============================] - 3s 38ms/step - loss: 1.1035 - sparse_categorical_accuracy: 0.6141
390/390 [==============================] - 32s 82ms/step - loss: 1.0825 - sparse_categorical_accuracy: 0.6291 - val_loss: 1.1035 - val_sparse_categorical_accuracy: 0.6141

Epoch 00011: LearningRateScheduler reducing learning rate to 0.00049999997.
Epoch 11/20
78/78 [==============================] - 4s 47ms/step - loss: 1.0661 - sparse_categorical_accuracy: 0.6295
390/390 [==============================] - 34s 88ms/step - loss: 1.0447 - sparse_categorical_accuracy: 0.6427 - val_loss: 1.0661 - val_sparse_categorical_accuracy: 0.6295

Epoch 00012: LearningRateScheduler reducing learning rate to 0.00042178275.
Epoch 12/20
78/78 [==============================] - 3s 39ms/step - loss: 1.0623 - sparse_categorical_accuracy: 0.6397
390/390 [==============================] - 33s 85ms/step - loss: 1.0299 - sparse_categorical_accuracy: 0.6485 - val_loss: 1.0623 - val_sparse_categorical_accuracy: 0.6397

Epoch 00013: LearningRateScheduler reducing learning rate to 0.00034549143.
Epoch 13/20
78/78 [==============================] - 3s 38ms/step - loss: 1.0278 - sparse_categorical_accuracy: 0.6429
390/390 [==============================] - 32s 83ms/step - loss: 0.9857 - sparse_categorical_accuracy: 0.6626 - val_loss: 1.0278 - val_sparse_categorical_accuracy: 0.6429

Epoch 00014: LearningRateScheduler reducing learning rate to 0.00027300484.
Epoch 14/20
78/78 [==============================] - 3s 39ms/step - loss: 1.0099 - sparse_categorical_accuracy: 0.6509
390/390 [==============================] - 34s 88ms/step - loss: 0.9505 - sparse_categorical_accuracy: 0.6762 - val_loss: 1.0099 - val_sparse_categorical_accuracy: 0.6509

Epoch 00015: LearningRateScheduler reducing learning rate to 0.00020610739.
Epoch 15/20
78/78 [==============================] - 3s 38ms/step - loss: 0.9996 - sparse_categorical_accuracy: 0.6556
390/390 [==============================] - 32s 82ms/step - loss: 0.9281 - sparse_categorical_accuracy: 0.6830 - val_loss: 0.9996 - val_sparse_categorical_accuracy: 0.6556

Epoch 00016: LearningRateScheduler reducing learning rate to 0.00014644662.
Epoch 16/20
78/78 [==============================] - 3s 39ms/step - loss: 0.9882 - sparse_categorical_accuracy: 0.6624
390/390 [==============================] - 34s 86ms/step - loss: 0.9009 - sparse_categorical_accuracy: 0.6910 - val_loss: 0.9882 - val_sparse_categorical_accuracy: 0.6624

Epoch 00017: LearningRateScheduler reducing learning rate to 9.54915e-05.
Epoch 17/20
78/78 [==============================] - 3s 39ms/step - loss: 0.9793 - sparse_categorical_accuracy: 0.6629
390/390 [==============================] - 32s 83ms/step - loss: 0.8856 - sparse_categorical_accuracy: 0.6970 - val_loss: 0.9793 - val_sparse_categorical_accuracy: 0.6629

Epoch 00018: LearningRateScheduler reducing learning rate to 5.4496708e-05.
Epoch 18/20
78/78 [==============================] - 4s 46ms/step - loss: 0.9761 - sparse_categorical_accuracy: 0.6631
390/390 [==============================] - 34s 86ms/step - loss: 0.8690 - sparse_categorical_accuracy: 0.7037 - val_loss: 0.9761 - val_sparse_categorical_accuracy: 0.6631

Epoch 00019: LearningRateScheduler reducing learning rate to 2.4471761e-05.
Epoch 19/20
78/78 [==============================] - 3s 39ms/step - loss: 0.9755 - sparse_categorical_accuracy: 0.6634
390/390 [==============================] - 34s 88ms/step - loss: 0.8644 - sparse_categorical_accuracy: 0.7069 - val_loss: 0.9755 - val_sparse_categorical_accuracy: 0.6634

Epoch 00020: LearningRateScheduler reducing learning rate to 6.155819e-06.
Epoch 20/20
78/78 [==============================] - 3s 38ms/step - loss: 0.9734 - sparse_categorical_accuracy: 0.6662
390/390 [==============================] - 32s 83ms/step - loss: 0.8573 - sparse_categorical_accuracy: 0.7080 - val_loss: 0.9734 - val_sparse_categorical_accuracy: 0.6662 
```
### Have a try
https://colab.research.google.com/drive/1ORdxhli85KzA-n5HalzmTmQWC2SFbfJg
