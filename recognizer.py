from sklearn import neighbors
from utils import convertToThreeChanneled, getEmbeddingsFromFile
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier

test_embeddings, test_labels = getEmbeddingsFromFile('embeddings/team_embeddings_v3.json')
knn = KNeighborsClassifier()
knn.fit(test_embeddings, test_labels)
class_index_lookup = {v:k for k,v in enumerate(knn.classes_)}

MODEL_PATH = "embeddings_models/team_data_models/embeddingModel_mobilenet_siamese_96valAccV2"
feature_extractor = load_model(MODEL_PATH, compile=False)

def preprocessImage(image_arr):
    image = Image.fromarray(image_arr)
    image = image.resize((128, 128))
    image = ImageOps.grayscale(image)
    image = np.array(image) / 255.0
    image = convertToThreeChanneled(image)
    return np.expand_dims(image, axis=0)

def recognizeClass(new_image, k=10):
    knn.n_neighbors = k
    preprocessed = preprocessImage(new_image)
    
    # Predict embedding from preprocessed image
    embedding = feature_extractor(preprocessed)
    y = knn.predict(embedding)[0]
    score = knn.predict_proba(embedding)[0][class_index_lookup[y]]
    return f"{y}[{score}]"   
    
    
