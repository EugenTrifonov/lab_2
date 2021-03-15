# 1.Нейронная сеть EfficientNet-B0 (случайное начальное приближение)
## 1)Структура
```python
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
## 2)Графики
![legend](https://user-images.githubusercontent.com/80068414/111124753-ea1d5b80-8581-11eb-8f4e-7cbae7714e62.png)

Метрика качества

![acc_1](https://github.com/EugenTrifonov/lab_2/blob/main/graphs/epoch_categorical_accuracy_1.svg)

Функция потерь

![loss_1](https://github.com/EugenTrifonov/lab_2/blob/main/graphs/epoch_loss_2.svg)
# С использованием техники обучения Transfer Learning  обучить нейронную сеть EfficientNet-B0 (предобученную на базе изображений imagenet)
## 1)Структура
## 2)Графики
