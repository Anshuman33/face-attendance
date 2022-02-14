import numpy as np
import json

def datagen_to_np_array(generator):
    images, labels = [], []
    generator.reset()
    for i in range(len(generator)):
        batch_X, batch_Y = generator.next()
        images.append(batch_X)
        labels.append(batch_Y)
    images = np.concatenate(images, axis = 0)
    # print(images.shape)
    if(images.shape[-1] == 1):
        images = np.repeat(images, 3, axis=-1)

    labels = np.concatenate(labels, axis = 0)
    labels = np.argmax(labels, axis = 1)    
    return images, labels

def saveEmbeddingFile(path, embeddings):
    with open(path, 'w') as jsonFile:
        json.dump(embeddings, jsonFile)
        
def getEmbeddingsFromFile(path):
    with open(path) as f:
        embeddings_dicts = json.load(f)
    embeddings = []
    labels = []
    for embedding_dict in embeddings_dicts:
        embeddings.append(embedding_dict['embedding'])
        labels.append(embedding_dict['class_label'])

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    return embeddings, labels

def convertToThreeChanneled(grayscale_batch):
    rgb_batch = np.repeat(grayscale_batch[..., np.newaxis],3,-1)
    return rgb_batch