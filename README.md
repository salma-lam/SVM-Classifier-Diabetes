# Classificateur SVM pour la Prédiction du Diabète

Ce mini projet implémente un classificateur à Support Vector Machine (SVM) pour prédire l'apparition du diabète à partir d'un ensemble de données de mesures diagnostiques médicales. La classification est réalisée à l'aide de SVM binaire et de SVM mono-classe pour la détection d'anomalies. Ce dépôt comprend le code Python pour l'entraînement, l'évaluation et la visualisation des modèles SVM avec différentes fonctions de noyau.

## Structure du Projet

- `diabetes.csv`: L'ensemble de données utilisé pour la classification, contenant des données médicales et des étiquettes de résultats.
- `binary_svm.py`: Implémentation de la classification SVM binaire pour prédire le diabète.
- `one_class_svm.py`: Implémentation de la classification SVM mono-classe pour la détection d'anomalies.
- `README.md`: Ce fichier, qui fournit une vue d'ensemble du projet.

## Installation

Pour exécuter ce projet, vous aurez besoin de Python et des bibliothèques suivantes :

- pandas
- scikit-learn
- matplotlib
- seaborn

Vous pouvez installer ces bibliothèques via pip :

```bash
pip install pandas scikit-learn matplotlib seaborn
