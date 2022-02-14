from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import datagen_to_np_array,saveEmbeddingFile
import numpy as np
import sys
import os

try:
    DATAPATH = sys.argv[1]
    EXTRACTOR_PATH = sys.argv[2]
    EMBEDDINGS_FILENAME = sys.argv[3]
except IndexError:
    exit("Please pass commandline arguments. Run the script by passing <dataset_path> <extractor_path> <target_filename>")
    
EMBEDDINGS_BASE_PATH = "embeddings/"

IMAGE_SIZE = (128, 128)
COLOR_MODE = "grayscale"
IMAGE_SHAPE = list(IMAGE_SIZE)+[3]


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
embeddings_extractor = load_model(EXTRACTOR_PATH, compile=False)
embeddings = embeddings_extractor(images).numpy()

# Save embeddings
print("Embedding successfully extracted. Shape:",embeddings.shape)
embeddings_list = [{'embedding':list(embedding.astype('float64')), 'class_label':str(label)} for embedding, label in zip(embeddings, labels)]

if not os.path.exists(EMBEDDINGS_BASE_PATH):
    os.makedirs(EMBEDDINGS_BASE_PATH)
save_path = os.path.join(EMBEDDINGS_BASE_PATH, EMBEDDINGS_FILENAME+".json") 
saveEmbeddingFile(save_path, embeddings_list)
print("Embeddings save successfully in", save_path)