import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os

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

def train():
    # Chargement des données
    df_train = pd.read_csv("img2000/train_full.csv")

    # Texte
    vectorizer = TfidfVectorizer(max_features=10000)
    X_text = vectorizer.fit_transform(df_train["description"].fillna("aucune description")).toarray()

    # Image
    base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    model = Model(inputs=base_model.input, outputs=base_model.output)

    img_features = []
    for img_id in df_train["imageid"]:
        img_path = os.path.join("img2000/image_train", f"{img_id}.jpg")
        img_features.append(extract_image_features(img_path, model))

    img_features = np.array(img_features)
    scaler = StandardScaler()
    pca = PCA(n_components=256)
    X_img = pca.fit_transform(scaler.fit_transform(img_features))

    # Fusion
    X = np.hstack((X_text, X_img))
    y = df_train["prdtypecode"]

    # Modèle
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    # Sauvegarde
    joblib.dump(clf, "models/classifier.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    joblib.dump(pca, "models/pca.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

if __name__ == "__main__":
    train()
