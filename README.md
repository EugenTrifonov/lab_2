# 1.Нейронная сеть EfficientNet-B0 (случайное начальное приближение)
## 1)Структура
```python
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
## 2)Графики

![acc_1](https://github.com/EugenTrifonov/lab_2/blob/main/graphs/epoch_categorical_accuracy_1.svg)

![loss_1](https://github.com/EugenTrifonov/lab_2/blob/main/graphs/epoch_loss_2.svg)
