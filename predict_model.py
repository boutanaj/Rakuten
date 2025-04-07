import numpy as np
import joblib
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model

def extract_image_features(image_path, model):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()
    except:
        return np.zeros((1280,))

def predict(description, image_path):
    # Chargement des objets
    clf = joblib.load("models/classifier.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")

    # Texte
    text_features = vectorizer.transform([description]).toarray()

    # Image
    base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    model = Model(inputs=base_model.input, outputs=base_model.output)
    img_features = extract_image_features(image_path, model)
    img_features_scaled = pca.transform(scaler.transform([img_features]))

    # Fusion
    X = np.hstack((text_features, img_features_scaled))

    # Prédiction
    return int(clf.predict(X)[0])
