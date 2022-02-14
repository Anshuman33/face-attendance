from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import datagen_to_np_array,saveEmbeddingFile
import numpy as np
import sys

DATAPATH = "cropped_images_team1/train/"
#model.summary()

IMAGE_SIZE = (128, 128)
COLOR_MODE = "grayscale"
IMAGE_SHAPE = list(IMAGE_SIZE)+[3]
NUM_CLASSES = 4

# Initialize generator
datagen = ImageDataGenerator(rescale=1./255,brightness_range=(1.0, 1.3))
generator = datagen.flow_from_directory(DATAPATH,
                                        color_mode=COLOR_MODE,
                                        target_size=IMAGE_SIZE, seed=10
                                    )
label_map = {v:k for k, v in generator.class_indices.items()}


# Load images
images, labels = datagen_to_np_array(generator)
labels = np.array(list(map(lambda x: label_map[x],labels)))
print("Images array shape: ", images.shape)
print("Labels array shape: ", labels.shape)


# Extract Embeddings
embeddings_extractor = load_model("models/team_embeddings_extractor_v2", compile=False)
embeddings = embeddings_extractor(images).numpy()

# Save embeddings
print("Embedding successfully extracted. Shape:",embeddings.shape)
embeddings_list = [{'embedding':list(embedding.astype('float64')), 'class_label':str(label)} for embedding, label in zip(embeddings, labels)]
EMBEDDINGS_PATH = "embeddings/team_embeddings_v2.json"
saveEmbeddingFile(EMBEDDINGS_PATH, embeddings_list)
print("Embeddings save successfully in",EMBEDDINGS_PATH)