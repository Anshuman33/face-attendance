from sklearn import neighbors
from utils import convertToThreeChanneled, getEmbeddingsFromFile
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier

test_embeddings, test_labels = getEmbeddingsFromFile('embeddings/house_embeddings.json')
knn = KNeighborsClassifier()
knn.fit(test_embeddings, test_labels)
class_index_lookup = {v:k for k,v in enumerate(knn.classes_)}


feature_extractor = load_model('models/house_embeddings_extractor', compile=False)

def preprocessImage(image_arr):
    image = Image.fromarray(image_arr)
    image = image.resize((128, 128))
    image = ImageOps.grayscale(image)
    image = np.array(image) / 255.0
    image = convertToThreeChanneled(image)
    return np.expand_dims(image, axis=0)

def getMajorityLabel(predLabels):
    c = Counter(predLabels)
    return c.most_common(1)[0][0]

def recognizeClass(new_image, k=5):
    knn.n_neighbors = k
    preprocessed = preprocessImage(new_image)
    
    # Predict embedding from preprocessed image
    embedding = feature_extractor(preprocessed)
    y = knn.predict(embedding)[0]
    score = knn.predict_proba(embedding)[0][class_index_lookup[y]]

    if (score >= (1-1/k)):
        return f"{y}[{score}]"
    else:
        return ""
    # Calculate distances from each existing embedding
    #distances = np.squeeze(euclidean_distances(embedding, test_embeddings))

    # Get the best distances
    # threshold = 0.2
    # labs = test_labels[distances < threshold]
    # print(labs)
    # sorted_dist = sorted(enumerate(distances), key=lambda x: x[1])
    #print(sorted_dist[:k])
    
    
    # if(np.mean(sorted_dist[:k]) < 0.3):
    #     labs = [test_labels[index] for index,dist in sorted_dist[:k]]
    #     return getMajorityLabel(labs)
    #print(sorted_dist)
    #return getMajorityLabel(labs)    
    
    
