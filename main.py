import os
import pathlib

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

# Datos de entrenamiento (MNIST, números escritos a mano)
# Cargar el conjunto de datos desde el directorio
dataset = tf.keras.utils.image_dataset_from_directory(
    'data',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(28, 28),
    batch_size=64
)

# Ruta al archivo .npy
file_path = 'data/full_numpy_bitmap_apple.npy'

# Cargar el archivo .npy
data = np.load(file_path)
# data a x_train
x_train = data

# Visualiza una imagen de ejemplo (puedes adaptar esto según tus necesidades)
imagen_de_ejemplo = data[0]
print(data.size)
matriz = np.reshape(imagen_de_ejemplo, (28, 28))
print(matriz.shape)
print(matriz)
plt.imshow(matriz, cmap='gray')  # Ajusta el mapa de colores según tu caso
plt.show()

#(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 127.5 - 1.0
x_train = x_train.reshape(x_train.shape[0], 784)


# Tamaño del espacio latente (z)
latent_dim = 100

# Construir el generador
generator = Sequential()
generator.add(Dense(256, input_dim=latent_dim, activation='relu'))
generator.add(Dense(784, activation='tanh'))

# Construir el discriminador
discriminator = Sequential()
discriminator.add(Dense(256, input_dim=784, activation='relu'))
discriminator.add(Dense(1, activation='sigmoid'))

# Compilar el discriminador
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Congelar el discriminador durante el entrenamiento del generador
discriminator.trainable = False

# Construir la GAN
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer='adam')


# Función para entrenar la GAN
def train_gan(gan, generator, discriminator, x_train, latent_dim, n_epochs=10000, batch_size=64):
    for epoch in range(n_epochs):
        # Entrenar el discriminador
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]
        labels = np.ones((batch_size, 1))
        fake_images = generator.predict(np.random.normal(0, 1, (batch_size, latent_dim)))
        d_loss_real = discriminator.train_on_batch(real_images, labels)
        labels = np.zeros((batch_size, 1))
        d_loss_fake = discriminator.train_on_batch(fake_images, labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Entrenar la GAN (generador)
        labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(np.random.normal(0, 1, (batch_size, latent_dim)), labels)

        # Mostrar el progreso
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

        # Guardar imágenes generadas
        if epoch % 1000 == 0:
            generate_and_save_images(generator, epoch)


# Función para generar y guardar imágenes generadas
def generate_and_save_images(model, epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    #save model
    model.save('model/generator_model.h5')
    model.save('model/discriminator_model.h5')
    gan.save('model/gan_model.h5')


    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = model.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i].reshape(28, 28), interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_generated_image_epoch_{epoch}.png')
    plt.show()


# Entrenar la GAN
train_gan(gan, generator, discriminator, x_train, latent_dim)
