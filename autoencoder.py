import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from keras.api.datasets import mnist
from keras.api.utils import to_categorical
from keras.api.layers import Activation
import matplotlib.pyplot as plt

# Окно для отображения цифр с примерами.
figure = plt.figure(figsize=(12, 4))
rows = 2
columns = 5

# Гиперпараметры.
EPOCHS = 10
BATCH_SIZE = 64
learning_rate = 0.001
latent_space = 128
loss = lambda x, y: tf.reduce_mean(tf.losses.binary_crossentropy(x, y))
opt = tf.optimizers.Adam(learning_rate=learning_rate)


# Полносвязный слой для encoder/decoder.
class Dense(tf.keras.layers.Layer):
    def __init__(self, units, activation, isEncoder=True):
        super().__init__()
        self.units = units
        self.activation = Activation(activation=activation)
        self.isEncoder = isEncoder

    def build(self, input):
        if self.isEncoder:
            self.w = self.add_weight(shape=(28 * 28, self.units), initializer='random_normal')
            self.b = self.add_weight(shape=(self.units,), initializer='zeros')
        else:
            self.w = self.add_weight(shape=(self.units, 28 * 28), initializer='random_normal')
            self.b = self.add_weight(shape=(28 * 28, ), initializer='zeros')

    def call(self, input):
        y = tf.matmul(input, self.w) + self.b
        return self.activation(y)


# Архитектура автокодировщика.
class MyAutoencoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.encoder = Dense(latent_space, "sigmoid")
        self.decoder = Dense(latent_space, "sigmoid", isEncoder=False)

    def __call__(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        return y


# Подгружаем и нормализуем данные из mnist.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.reshape(x_train, [-1, 28 * 28])
x_test = tf.reshape(x_test, [-1,  28 * 28])
x_train /= 255
x_test /= 255

# Сразу соеднияем и перемешиваем входное значение и правильный ответ.
# Естественно для автокодировщика входное значение и выходное это должно быть одно и тоже.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

# Создаем класс с нейронной сетью.
modul = MyAutoencoder()

# Функция которая находит градиент, ошибку для одного батча. А также используется метод БП.
@tf.function
def train_batch(x_batch, y_batch):
    with tf.GradientTape() as tape:
        f_loss = loss(y_batch, modul(x_batch))

    grad = tape.gradient(f_loss, modul.trainable_variables)
    opt.apply_gradients(zip(grad, modul.trainable_variables))

    return f_loss

# Цикл самого обучения.
for n in range(EPOCHS):
    losses = 0
    for x, y in train_dataset:
        losses += train_batch(x, y)

    print(losses.numpy())

# Кол-во картинок В окне для примера
img_count = columns

# Показ цифр из тестового множеста.
for i in range(img_count):
    figure.add_subplot(rows, columns, i + 1)
    plt.imshow(tf.reshape(x_test[i], (28, 28, 1)), cmap="gray")

for i in range(img_count):
    figure.add_subplot(rows, columns, i + 1 + img_count)
    plt.imshow(tf.reshape(modul(x_test[i:i+1]), (28, 28, 1)), cmap="gray")

plt.show()

