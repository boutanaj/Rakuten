import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ✅ Chargement des données
print("Chargement des données...")
df_Xtrain = pd.read_csv("img2000/X_train.csv")
df_Xtest = pd.read_csv("img2000/X_test.csv")
df_Ytrain = pd.read_csv("img2000/Y_train.csv")
df_Ytest = pd.read_csv("img2000/Y_test.csv")
print("Données chargées avec succès !")

# ✅ Traitement des valeurs manquantes
print("Traitement des valeurs manquantes...")
df_Xtrain['description'] = df_Xtrain['description'].fillna("aucune description")
df_Xtest['description'] = df_Xtest['description'].fillna("aucune description")

# ✅ Fusion des features avec les labels
print("Fusion des datasets...")
df_train = df_Xtrain.merge(df_Ytrain, on="Unnamed: 0").drop(columns=["Unnamed: 0"])
df_test = df_Xtest.merge(df_Ytest, on="Unnamed: 0").drop(columns=["Unnamed: 0"])

# ✅ Extraction des features textuelles
print("Vectorisation des descriptions textuelles...")
vectorizer = TfidfVectorizer(max_features=10000)
X_train_text = vectorizer.fit_transform(df_train["description"]).toarray()
X_test_text = vectorizer.transform(df_test["description"]).toarray()

# ✅ Chargement du modèle de feature extraction d'images
print("Chargement du modèle CNN pour l'extraction des features...")
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# ✅ Fonction d'extraction des features d'une image
def extract_image_features(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()
    except:
        return np.zeros((1280,))

# ✅ Chargement des features des images
def get_image_features(df, image_folder):
    print(f"Extraction des features pour {len(df)} images...")
    image_features = []
    zero_count = 0
    for img_id in df["imageid"]:
        img_path = os.path.join(image_folder, f"{img_id}.jpg")
        features = extract_image_features(img_path)
        if np.all(features == 0):
            zero_count += 1
        image_features.append(features)
    print(f"Nombre de vecteurs nuls : {zero_count}")
    return np.array(image_features)

# ✅ Extraction des features d'images
X_train_img = get_image_features(df_train, "img2000/image_train")
X_test_img = get_image_features(df_test, "img2000/image_test")

# ✅ Réduction de dimension des features d'images (optionnel)
print("Réduction de dimension des features d'images...")
scaler = StandardScaler()
pca = PCA(n_components=256)  # On réduit à 256 dimensions
X_train_img = pca.fit_transform(scaler.fit_transform(X_train_img))
X_test_img = pca.transform(scaler.transform(X_test_img))

# ✅ Fusion des features textuelles et visuelles
print("Fusion des features texte et image...")
X_train = np.hstack((X_train_text, X_train_img))
X_test = np.hstack((X_test_text, X_test_img))

# ✅ Définition du modèle de classification
print("Création du pipeline et début de l'entraînement...")
classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

# ✅ Entraînement du modèle
start_time = time.time()
classifier.fit(X_train, df_train["prdtypecode"])
training_time = time.time() - start_time
print(f"Modèle entraîné en {training_time:.2f} secondes.")

# ✅ Prédictions et évaluation
print("Prédiction et évaluation du modèle...")
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(df_test["prdtypecode"], y_pred)
print(f"Précision du modèle : {accuracy:.4f}")
