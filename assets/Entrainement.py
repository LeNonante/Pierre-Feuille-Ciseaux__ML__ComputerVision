import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données
data = pd.read_csv("donnees.csv")

# Séparer les caractéristiques (X) et les étiquettes (y)
X = data.iloc[:, 1:].values  # Toutes les colonnes sauf la première (les coordonnées des landmarks)
y = data.iloc[:, 0].values  # La première colonne (étiquettes)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Évaluer le modèle
y_pred = clf.predict(X_test)
print("Précision :", accuracy_score(y_test, y_pred))

# Sauvegarder le modèle pour utilisation future
import joblib
joblib.dump(clf, "gesture_model.pkl")