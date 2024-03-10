import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Cargar el modelo generador
generator = load_model('model/generator_model.h5')

# Generar ruido en el espacio latente
latent_dim = 100
noise = np.random.normal(0, 1, (20, latent_dim))

# Utilizar el generador para generar imágenes a partir del ruido
generated_images = generator.predict(noise)

# Ajustar la escala de las imágenes generadas a [0, 1]
generated_images = 0.5 * generated_images + 0.5

# Mostrar las imágenes generadas
plt.figure(figsize=(20, 1))
for i in range(20):
    plt.subplot(1, 20, i + 1)
    plt.imshow(generated_images[i].reshape(28, 28), interpolation='nearest', cmap='gray')
    plt.axis('off')
plt.show()