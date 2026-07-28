# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv (3.10.12)
#     language: python
#     name: python3
# ---

# # Autoencoder
#
# ## Wie funktioniert ein Autoencoder?
#
# Ein **Autoencoder** ist ein neuronales Netzwerk, das lernt, Daten zu komprimieren und wieder zu rekonstruieren. Die Grundidee besteht darin, ein neuronales Netz so zu trainieren, dass es eine Eingabe zunächst in eine niedrigdimensionale Darstellung (Kodierung) überführt und anschließend aus dieser komprimierten Darstellung die ursprüngliche Eingabe wiederherstellt (Dekodierung).
#
# Mit anderen Worten: Autoencoders sind ein Verfahren des unüberwachten Lernens, das lernt, Daten zu komprimieren, indem eine kompakte Repräsentation der Eingabedaten (die Kodierung) erstellt wird und daraus die ursprünglichen Daten wieder rekonstruiert werden (die Dekodierung).
#
# Autoencoder sind für viele Aufgaben geeignet, zum Beispiel für Dimensionsreduktion, Datenkompression und Rauschunterdrückung. Sie werden außerdem in der Bild- und Spracherkennung sowie der Verarbeitung natürlicher Sprache eingesetzt.
#
# Autoencoder werden mit Backpropagation trainiert, wobei die Gewichte des neuronalen Netzes so angepasst werden, dass der Unterschied zwischen der ursprünglichen Eingabe und der rekonstruierten Ausgabe minimiert wird. Es gibt verschiedene Varianten von Autoencodern, darunter einfache Autoencoder, Denoising Autoencoder, Variational Autoencoder und Convolutional Autoencoder, jeder mit eigenen Architekturen und Anwendungsfällen.
#
# ### Einfacher Autoencoder
#
# Ein einfacher Autoencoder könnte so aussehen:
#
# ![einfacher_Autoencoder.png](Bilder/einfacher_Autoencoder.png)

# +
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define input size
input_size = 784  # 28x28

# Define encoding and decoding size
encoding_size = 64

# Define input layer
input_layer = Input(shape=(input_size,))

# Define encoding layer
encoding_layer = Dense(encoding_size, activation='relu')(input_layer)

# Define decoding layer
decoding_layer = Dense(input_size, activation='sigmoid')(encoding_layer)

# Create the autoencoder model
autoencoder = Model(input_layer, decoding_layer)

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Print the model summary
autoencoder.summary()
# -

# In diesem Beispiel wird ein einfacher Autoencoder mit einer einzelnen verdeckten Schicht aus 64 Neuronen erstellt. Er kodiert ein Eingabebild der Größe 784 (28×28 Pixel) in eine kleinere Darstellung und dekodiert es anschließend wieder zum ursprünglichen Bild. Als Verlustfunktion wird die binäre Kreuzentropie und als Optimierer Adam verwendet. Nach dem Erstellen und Kompilieren des Modells kann es mit der Methode `fit()` auf den Trainingsdaten trainiert werden.
#

# +
## Daten einlesen

#| code-fold: true
#| code-summary: "Code anzeigen"


import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re
import cv2

# Funktion zur numerischen Sortierung der Dateien
def extract_number(filename):
    return int(re.search(r"_sortiert(\d+)", filename).group(1))

# Alle Files einleden, die mit Zahlen_Handschrift_sortiert beginnen und direkt darauf folgend eine Zahl im Namen haben.
liste_daten = glob('Daten/UeberwachtesLernen/Zahlen_Handschrift_sortiert*[0-9].png')

# Liste numerisch sortieren und sortierte Liste ausgeben
liste_daten = sorted(liste_daten, key=extract_number)
for el in liste_daten:
    print(el)

#| code-fold: true
#| code-summary: "Code anzeigen"


def imshow(name, img, size=8):
    import matplotlib.pyplot as plt
    from IPython.display import display
    import cv2

    # Convert BGR to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    aspect = w / h
    plt.figure(figsize=(size * aspect, size))
    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    plt.title(name)
    plt.axis('off')
    #plt.show()

# contours are not sorted, so sort them to assign the correct labels later
def sort_contours_grid(contours, row_height=50):
    # Extract bounding boxes
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    # Combine contours with their boxes
    contours_with_boxes = list(zip(contours, bounding_boxes))

    # First sort top-to-bottom by y
    contours_with_boxes.sort(key=lambda b: b[1][1])  # b[1][1] = y

    # Group contours into rows based on y
    rows = []
    current_row = []
    last_y = -row_height

    for c, (x, y, w, h) in contours_with_boxes:
        if y - last_y > row_height:
            if current_row:
                rows.append(current_row)
            current_row = [(c, (x, y, w, h))]
            last_y = y
        else:
            current_row.append((c, (x, y, w, h)))
    if current_row:
        rows.append(current_row)

    # Now sort each row left to right
    sorted_contours = []
    for row in rows:
        row.sort(key=lambda b: b[1][0])  # sort by x
        sorted_contours.extend([c for c, _ in row])

    return sorted_contours

#| code-fold: true
#| code-summary: "Code anzeigen"

contour_height = 23

x_data_list = []
response_list = []

## Alle Bilder auswerten
for im in liste_daten:

    print(im)
    image = cv2.imread(im)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    # find digits in image and create contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    no_digits = 0
    #plt.figure(figsize = (10,15))
    #plt.imshow(image, cmap='gray')
    #plt.title(im)

    # jetzt rotes Viereck um jede Zahl zeichnen
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > contour_height:
                no_digits +=1
    #            plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2))

    #Sortieren der Konturen von oben nach unten und links nach rechts
    contours = sort_contours_grid(contours)

    for cnt in contours:
        # print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > contour_height:
                roi = thresh[y:y+h, x:x+w]
                roismall = cv2.resize(roi, (28, 28))
                x_data_list.append(roismall)

    # now generate labels
    print(no_digits, no_digits/10)
    # create list of labels
    list_of_digits = [0,1,2,3,4,5,6,7,8,9]
    
    for element in range(int(no_digits/10)):
        response_list.append(list_of_digits)

#finalize the data
x_data = np.array(x_data_list)
response = np.array(response_list)
y_data = response.flatten()

# +
#HIER!

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Daten vorbereiten: normieren und zu Vektoren umformen
x = x_data.astype('float32') / 255.
x = x.reshape(-1, 784)

# Trainings-/Test-Split
split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y_data[:split], y_data[split:]

# Modell definieren
input_size   = 784
encoding_size = 64

input_layer    = Input(shape=(input_size,))
encoding_layer = Dense(encoding_size, activation='relu')(input_layer)
decoding_layer = Dense(input_size,    activation='sigmoid')(encoding_layer)

autoencoder = Model(input_layer, decoding_layer)
encoder     = Model(input_layer, encoding_layer)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

# Training
history = autoencoder.fit(
    x_train, x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Trainingsverlauf
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'],     label='Training')
plt.plot(history.history['val_loss'], label='Validierung')
plt.xlabel('Epoche')
plt.ylabel('Verlust')
plt.title('Trainingsverlauf')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Rekonstruktionen visualisieren
n = 10
x_sample        = x_test[:n]
rekonstruktionen = autoencoder.predict(x_sample)

plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), cmap='gray')
    plt.title(f'Original ({y_test[i]})')
    plt.axis('off')

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(rekonstruktionen[i].reshape(28, 28), cmap='gray')
    plt.title('Rekonstruktion')
    plt.axis('off')

plt.tight_layout()
plt.show()

# -

# ## Variational Autoencoder (VAE)
#
# Ein normaler Autoencoder komprimiert jede Eingabe auf einen **festen Punkt** im latenten Raum. Das Problem: Zwischen zwei gelernten Punkten liegt oft nichts Sinnvolles – der latente Raum hat "Lücken".
#
# Ein **Variational Autoencoder (VAE)** löst das, indem er statt eines festen Punktes eine **Wahrscheinlichkeitsverteilung** lernt:
#
# - Der Encoder liefert nicht mehr *einen* Punkt $h$, sondern einen **Mittelwert** $\mu$ und eine **Streuung** $\sigma$.
# - Daraus wird $h$ zufällig gezogen: $h \sim \mathcal{N}(\mu, \sigma)$
# - Der Decoder muss trotzdem korrekt rekonstruieren – auch wenn $h$ leicht variiert.
#
# Das zwingt das Netz dazu, den latenten Raum **zusammenhängend und geordnet** zu halten. Benachbarte Punkte im latenten Raum ergeben ähnliche Ausgaben.
#
# ### Verlustfunktion
#
# Die Verlustfunktion besteht aus zwei Teilen:
#
# $$L = \underbrace{\|x - r\|^2}_{\text{Rekonstruktionsfehler}} + \underbrace{D_{KL}\big(\mathcal{N}(\mu,\sigma) \,\|\, \mathcal{N}(0,1)\big)}_{\text{KL-Divergenz}}$$
#
# | Teil | Bedeutung |
# |---|---|
# | Rekonstruktionsfehler | Original und Rekonstruktion sollen möglichst ähnlich sein |
# | KL-Divergenz | Die gelernte Verteilung soll nah an der Standardnormalverteilung bleiben |
#
# Die KL-Divergenz verhindert, dass der Encoder die Verteilungen zu weit auseinanderzieht und sorgt so dafür, dass der latente Raum "lückenlos" bleibt.
#
# ### Warum ist das nützlich?
#
# Da der gesamte latente Raum sinnvoll befüllt ist, kann man **neue Daten generieren**: Man zieht einen zufälligen Punkt aus $\mathcal{N}(0,1)$ und schickt ihn durch den Decoder – das Ergebnis ist ein neues, plausibles Bild.
#
# Das folgende Beispiel trainiert einen VAE auf dem MNIST-Datensatz und erzeugt neue Ziffernbilder:
#

# +
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampling((z_mean, z_log_var))
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, data):
        z_mean, z_log_var = self.encoder(data)
        z = self.sampling((z_mean, z_log_var))
        return self.decoder(z)

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Normalize image data
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

vae.fit(x_train, epochs=30, batch_size=128)

# Generate new images from the learned latent space
n = 10
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# Sample random vectors from a normal distribution
z_sample = np.random.normal(size=(n * n, latent_dim))
# Decode the vectors into images
x_decoded = decoder.predict(z_sample)
# Reshape the images and combine them into a grid
for i in range(n):
    for j in range(n):
        digit = x_decoded[i * n + j].reshape(digit_size, digit_size)
        figure[i * digit_size : (i + 1) * digit_size,
               j * digit_size : (j + 1) * digit_size] = digit
# Plot the grid of images
plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap="Greys_r")
plt.axis("off")
plt.show()

#| code-fold: true
#| code-summary: "Code anzeigen"

contour_height = 23

x_data_list = []
response_list = []

## Für die bessere Übersicht im Skript werden hier nur die ersten zwei Bilder ausgewertet. Wenn alle Bilder ausgewertet werden sollen, kommende Zeile auskommentieren und die alternative for-Schleife reinnehmen.
#for im in liste_daten[:2]:

## Alle Bilder auswerten
for im in liste_daten:

    print(im)
    image = cv2.imread(im)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    # find digits in image and create contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    no_digits = 0
    plt.figure(figsize = (10,15))
    plt.imshow(image, cmap='gray')
    plt.title(im)

    # jetzt rotes Viereck um jede Zahl zeichnen
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > contour_height:
                no_digits +=1
                plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2))

    #Sortieren der Konturen von oben nach unten und links nach rechts
    contours = sort_contours_grid(contours)

    for cnt in contours:
        # print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > contour_height:
                roi = thresh[y:y+h, x:x+w]
                roismall = cv2.resize(roi, (28, 28))
                x_data_list.append(roismall)

    # now generate labels
    print(no_digits, no_digits/10)
    # create list of labels
    list_of_digits = [0,1,2,3,4,5,6,7,8,9]
    
    for element in range(int(no_digits/10)):
        response_list.append(list_of_digits)

#finalize the data
x_data = np.array(x_data_list)
response = np.array(response_list)
y_data = response.flatten()
# -

# ## Undercomplete Autoencoder
#
# Ein **undercomplete Autoencoder** ist ein Autoencoder, der lernt, Eingabedaten in einen niedrigdimensionalen latenten Raum zu komprimieren. Die Dimension des latenten Raums ist dabei kleiner als die Dimension der Eingabedaten, was einen Flaschenhals (*Bottleneck*) erzeugt und den Autoencoder dazu zwingt, eine kompaktere Darstellung der Daten zu erlernen.
#
# Eine Möglichkeit, einen undercomplete Autoencoder zu erstellen, besteht darin, eine Flaschenhalsschicht mit weniger Neuronen als die Eingabe- und Ausgabeschichten zu verwenden. Der Encoder bildet die Eingabedaten durch diese Schicht auf den latenten Raum ab, und der Decoder bildet den latenten Raum wieder auf die Ausgabedaten zurück.
#
# Hier ist ein Beispiel der Implementierung eines undercomplete Autoencoders in Python mit Keras und TensorFlow auf dem MNIST-Datensatz:
#

# +
import numpy as np
from tensorflow import keras

# Load the MNIST dataset
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Normalize the data and reshape it
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], -1)

# Set the dimensionality of the encoding space
encoding_dim = 32

# Define the input layer
input_img = keras.Input(shape=(784,))

# Define the encoded layer
encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_img)

# Define the decoded layer
decoded = keras.layers.Dense(784, activation='sigmoid')(encoded)

# Define the autoencoder model
autoencoder = keras.Model(input_img, decoded)

# Define the encoder model
encoder = keras.Model(input_img, encoded)

# Define the decoder model
decoder_input = keras.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(decoder_input, decoder_layer(decoder_input))

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder model
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Encode and decode some digits from the test set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
# -

# ## Sparse Autoencoder
#
# Ein **Sparse Autoencoder** ist ein Autoencoder, der darauf ausgelegt ist, eine komprimierte Darstellung der Eingabedaten zu erlernen und dabei Sparsität (*Dünnbesetztheit*) in der gelernten Repräsentation zu erzwingen. Die Sparsitätsbedingung fördert, dass der Autoencoder eine Darstellung lernt, die nur eine geringe Anzahl von Neuronen in der verdeckten Schicht zur Kodierung der Eingabe verwendet. Dies kann dabei helfen, Overfitting zu reduzieren und die Interpretierbarkeit der gelernten Repräsentation zu verbessern.
#
# Beim Sparse Autoencoder wird die Sparsitätsbedingung typischerweise durch einen Regularisierungsterm in der Verlustfunktion erzwungen, der Aktivierungen der verdeckten Schicht bestraft, die deutlich von einem kleinen Zielwert abweichen.
#
# Hier ist ein Beispiel der Implementierung eines Sparse Autoencoders in Python mit Keras und TensorFlow:
#

# +
import numpy as np
from tensorflow import keras

# Load the MNIST dataset
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Normalize the data and reshape it
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], -1)

# Set the dimensionality of the encoding space
encoding_dim = 32

# Define the input layer
input_img = keras.Input(shape=(784,))

# Define the encoded layer with a sparsity constraint
encoded = keras.layers.Dense(encoding_dim, activation='relu',
                             activity_regularizer=keras.regularizers.l1(10e-5))(input_img)

# Define the decoded layer
decoded = keras.layers.Dense(784, activation='sigmoid')(encoded)

# Define the autoencoder model
autoencoder = keras.Model(input_img, decoded)

# Define the encoder model
encoder = keras.Model(input_img, encoded)

# Define the decoder model
decoder_input = keras.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(decoder_input, decoder_layer(decoder_input))

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder model
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Encode and decode some digits from the test set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
# -

# ## Denoising Autoencoder
#
# Ein **Denoising Autoencoder** ist ein Autoencoder, der darauf ausgelegt ist, Rauschen aus den Eingabedaten zu entfernen. Er wird auf verrauschten Versionen der Eingabedaten trainiert, wobei das Rauschen auf verschiedene Weisen eingebracht werden kann, z.B. durch Hinzufügen von Gaußschem Rauschen, das zufällige Maskieren von Pixeln oder das Anwenden zufälliger Rotationen auf die Eingabedaten. Der Autoencoder lernt dann, die ursprünglichen Eingabedaten aus der verrauschten Version zu rekonstruieren, was dazu genutzt werden kann, das Rauschen aus neuen, unbekannten Daten zu entfernen.
#
# Hier ist ein Beispiel der Implementierung eines Denoising Autoencoders in Python mit Keras und TensorFlow:
#

# +
import numpy as np
from tensorflow import keras

# Load the MNIST dataset
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Normalize the data and reshape it
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], -1)

# Add random Gaussian noise to the input data
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip the noisy data to be between 0 and 1
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Set the dimensionality of the encoding space
encoding_dim = 32

# Define the input layer
input_img = keras.Input(shape=(784,))

# Define the encoded layer
encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_img)

# Define the decoded layer
decoded = keras.layers.Dense(784, activation='sigmoid')(encoded)

# Define the denoising autoencoder model
autoencoder = keras.Model(input_img, decoded)

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder model
autoencoder.fit(x_train_noisy, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

# Use the autoencoder to remove noise from some digits in the test set
decoded_imgs = autoencoder.predict(x_test_noisy)
# -

# In diesem Beispiel wird den Eingabedaten Gaußsches Rauschen mit einem Faktor von 0,5 hinzugefügt, was eine verrauschte Version der Originaldaten erzeugt. Anschließend wird ein Denoising Autoencoder trainiert, der die Originaldaten aus der verrauschten Version rekonstruiert. Die restliche Implementierung ähnelt der eines regulären Autoencoders. Nach dem Training kann der Denoising Autoencoder genutzt werden, um Rauschen aus Testdaten zu entfernen, indem die verrauschten Daten durch den Autoencoder geleitet und die rekonstruierte Ausgabe erhalten wird.
#

# ## Contractive Autoencoder
#
# Ein **Contractive Autoencoder** ist ein Autoencoder, der darauf ausgelegt ist, eine robustere Repräsentation der Eingabedaten zu erlernen, indem ein zusätzlicher Regularisierungsterm zur Verlustfunktion hinzugefügt wird. Dieser Term fördert, dass der Autoencoder eine komprimierte Darstellung der Eingabedaten erlernt, die gegenüber kleinen Störungen im Eingaberaum invariant ist. Dies wird erreicht, indem die Ableitung der Encoder-Ausgabe bezüglich der Eingabe bestraft wird.
#
# Hier ist ein Beispiel der Implementierung eines Contractive Autoencoders in Python mit Keras und TensorFlow:
#

# +
import numpy as np
from tensorflow import keras

# Load the MNIST dataset
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Normalize the data and reshape it
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

encoding_dim = 32
lam = 1e-4

# Build model
input_img = keras.Input(shape=(784,))
encoder_layer = keras.layers.Dense(encoding_dim, activation="sigmoid", name="encoder")
encoded = encoder_layer(input_img)
decoded = keras.layers.Dense(784, activation="sigmoid", name="decoder")(encoded)

autoencoder = keras.Model(input_img, decoded)

# Custom loss
def contractive_loss(y_true, y_pred):
    # Reconstruction loss per sample
    mse = keras.ops.mean(keras.ops.square(y_true - y_pred), axis=1)

    # Encoder weights: shape (784, encoding_dim)
    W = encoder_layer.kernel

    # Hidden activations for the current batch
    h = encoder_layer(y_true)

    # Derivative of sigmoid activation
    dh = h * (1.0 - h)

    # Sum of squared weights per hidden unit
    W_squared_sum = keras.ops.sum(keras.ops.square(W), axis=0)

    # Contractive penalty per sample
    contractive = lam * keras.ops.sum(keras.ops.square(dh) * W_squared_sum, axis=1)

    return mse + contractive

autoencoder.compile(optimizer="adam", loss=contractive_loss)

autoencoder.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
)

decoded_imgs = autoencoder.predict(x_test)
# -

print(decoded_imgs.shape)

# ## Anwendungsfälle von Autoencoders
#
# ### Datenkompression
#
# Autoencoders werden häufig zur Datenkompression eingesetzt. Dabei wird der Autoencoder darauf trainiert, eine komprimierte Darstellung der Eingabedaten zu erlernen. Die komprimierte Darstellung kann dann mit geringerem Speicherbedarf und weniger Bandbreite gespeichert oder übertragen werden. Dieser Anwendungsfall ist besonders relevant für große Datensätze, bei denen eine effiziente Speicherung und Übertragung entscheidend ist.
#
# ### Dimensionsreduktion
#
# Autoencoders können zur Dimensionsreduktion verwendet werden, wobei die Anzahl der Merkmale oder Variablen in einem Datensatz reduziert wird. Dies kann helfen, die Komplexität eines Modells zu reduzieren oder die Daten zu vereinfachen. Durch die Komprimierung der Eingabedaten in eine niedrigdimensionale Darstellung können Autoencoders dabei helfen, die wichtigsten Merkmale oder Variablen in einem Datensatz zu identifizieren und so die Analyse und Visualisierung zu erleichtern.
#
# ### Anomalieerkennung
#
# Autoencoders können zur Anomalieerkennung eingesetzt werden, bei der das Ziel darin besteht, Datenpunkte zu identifizieren, die sich deutlich vom Rest des Datensatzes unterscheiden. Durch das Training eines Autoencoders auf einem Datensatz kann das Netz lernen, die typischen Muster des Datensatzes zu erkennen. Bei einem neuen Datenpunkt kann der Autoencoder diesen mit den gelernten Mustern vergleichen und feststellen, ob es sich um eine Anomalie handelt.
#
# ### Bild- und Audiorekonstruktion
#
# Autoencoders können auch zur Bild- und Audiorekonstruktion eingesetzt werden. Durch das Training eines Autoencoders auf einem Datensatz aus Bildern oder Audioaufnahmen kann das Netz lernen, die Eingabedaten aus einer komprimierten Darstellung zu rekonstruieren. Dies kann für Aufgaben wie die Bild- und Audiorauschunterdrückung nützlich sein, bei der das Ziel darin besteht, Rauschen aus den Eingabedaten zu entfernen.
#
# ### Datengenerierung
#
# Schließlich können Autoencoders zur Datengenerierung verwendet werden, bei der das Ziel darin besteht, neue Datenpunkte zu erzeugen, die den Trainingsdaten ähneln. Durch das Training eines Autoencoders auf einem Datensatz kann das Netz lernen, neue Datenpunkte zu generieren, die ähnliche Merkmale oder Eigenschaften aufweisen.
#
