from utils import convertToThreeChanneled, getEmbeddingsFromFile
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from sklearn.neighbors import KNeighborsClassifier

test_embeddings, test_labels = getEmbeddingsFromFile('embeddings/team_embeddings_v4.json')
knn = KNeighborsClassifier(weights="distance")
knn.fit(test_embeddings, test_labels)
class_index_lookup = {v:k for k,v in enumerate(knn.classes_)}

MODEL_PATH = "embeddings_models/team_data_models/embeddingModel_mobilenet_siamese_98Acc"
feature_extractor = load_model(MODEL_PATH, compile=False)

def preprocessImage(image_arr):
    image = Image.fromarray(image_arr)
    image = image.resize((128, 128))
    image = ImageOps.grayscale(image)
    image = np.array(image) / 255.0
    image = convertToThreeChanneled(image)
    return np.expand_dims(image, axis=0)

def recognizeClass(new_image, k=5):
    knn.n_neighbors = k
    preprocessed = preprocessImage(new_image)
    
    # Predict embedding from preprocessed image
    embedding = feature_extractor(preprocessed)
    y = knn.predict(embedding)[0]
    score = knn.predict_proba(embedding)[0][class_index_lookup[y]]
    return f"{y} [{score:.2f}]"   
    
    
