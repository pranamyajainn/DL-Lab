# Q5: Implement image style transfer using a pre-trained model (one-by-one image display)

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Load and preprocess the image
def load_and_process_image(image_path):
    img = tf.image.decode_image(tf.io.read_file(image_path), channels=3)
    img = tf.image.resize(img, [512, 512])
    return img[tf.newaxis, ...] / 255.0  # normalize and add batch dimension

# Load pre-trained model from TensorFlow Hub
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Load content and style images
content_image = load_and_process_image('content.jpeg')
style_image = load_and_process_image('style.jpeg')

# Apply style transfer
stylized_image = model(content_image, style_image)[0]

# Show images one by one with titles
titles = ['Content Image', 'Style Image', 'Stylized Image']
images = [content_image, style_image, stylized_image]

for i in range(3):
    plt.imshow(images[i][0])
    plt.title(titles[i])
    plt.axis('off')
    plt.show()
