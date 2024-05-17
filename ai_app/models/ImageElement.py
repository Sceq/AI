# ai_app/models/ImageElement.py

from django.db import models
from django.core.files.storage import default_storage
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as tf_image
import webcolors

def get_dominant_colors(image_path, n_colors=3):
    image = Image.open(image_path)
    image = image.resize((100, 100)) # zmniejszam wielkość obrazu - szybciej przetworzy obraz.
    image_np = np.array(image)
    image_np = image_np.reshape((image_np.shape[0] * image_np.shape[1], 3))

    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(image_np)
    colors = kmeans.cluster_centers_
    return colors.astype(int).tolist()

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(rgb_color):
    try:
        return webcolors.rgb_to_name(rgb_color)
    except ValueError:
        return closest_color(rgb_color)

class ImageElement(models.Model):
    title = models.CharField(max_length=100, blank=True)
    content = models.TextField(blank=True)
    photo = models.ImageField(upload_to='mediaphoto', blank=True, null=True)
    dominant_colors = models.JSONField(blank=True, null=True)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        if self.photo:
            try:
                file_path = self.photo.path
                if default_storage.exists(file_path):
                    pill_image = tf_image.load_img(file_path, target_size=(299, 299))
                    img_array = tf_image.img_to_array(pill_image)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    model = InceptionV3(weights='imagenet')
                    prediction = model.predict(img_array)
                    decoded_prediction = decode_predictions(prediction, top=1)[0]
                    best_guess = decoded_prediction[0][1]
                    self.title = best_guess
                    self.content = ', '.join([f"{pred[1]}: {pred[2] * 100:.2f}%" for pred in decoded_prediction])

                    colors_rgb = get_dominant_colors(file_path)
                    colors_names = [get_color_name(color) for color in colors_rgb]
                    self.dominant_colors = colors_names

                    super().save(*args, **kwargs)

            except Exception as e:
                print(e)
                pass
