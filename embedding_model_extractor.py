from tensorflow.keras.models import load_model
import keras.backend as K
import sys
import os
SIAMESE_MODELS_BASE_PATH = "siamese_models/"
EMBEDDING_MODELS_BASE_PATH = "embeddings_models/"

# Wrapper function to return the contastive loss function with margin
def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        pred_square = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * pred_square + (1-y_true)*margin_square)
    return contrastive_loss 

try:
    siamese_path = sys.argv[1]
except IndexError:
    exit("Please provide siamese model path.")

prefixPath, modelName = os.path.split(siamese_path)

# Load the siamese network model
model = load_model(os.path.join(SIAMESE_MODELS_BASE_PATH, siamese_path),
             custom_objects={'contrastive_loss':contrastive_loss_with_margin(1.05)}                                                        
            )

print("Loaded the model successfully. Model Summary:-")
model.summary()

# Get the feature extractor layer from the network
embeddings_model = model.get_layer('base_network')
embeddings_model.save(os.path.join(EMBEDDING_MODELS_BASE_PATH, prefixPath, "embeddingModel_"+modelName),
                      include_optimizer=False)

print("Embeddings Model Saved successfully.")