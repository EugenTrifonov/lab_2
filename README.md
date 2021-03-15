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

![loss_1](https://github.com/EugenTrifonov/lab_2/blob/main/graphs/epoch_loss_1.svg)
# 2.С использованием техники обучения Transfer Learning  обучить нейронную сеть EfficientNet-B0 (предобученную на базе изображений imagenet)
## Попытка 1
## 1)Структура
```python
inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
model = EfficientNetB0(include_top=False,input_tensor=inputs, weights="imagenet")
model.trainable=False
x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
return tf.keras.Model(inputs=inputs, outputs=outputs)
```
## 2)Графики
![legend](https://user-images.githubusercontent.com/80068414/111124753-ea1d5b80-8581-11eb-8f4e-7cbae7714e62.png)

Метрика качества

![acc_2](https://github.com/EugenTrifonov/lab_2/blob/main/graphs/epoch_categorical_accuracy_2.svg)

Функция потерь

![loss_2](https://github.com/EugenTrifonov/lab_2/blob/main/graphs/epoch_loss_2.svg)

## Попытка 2
Поскольку в первом случае результат был неудовлетворительным, было принято решение адаптировать код с https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/  
## 1)Структура
Файл https://github.com/EugenTrifonov/lab_2/blob/main/train_new.py
## 2)Графики
