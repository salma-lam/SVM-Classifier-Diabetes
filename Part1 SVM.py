                                                 #TP SVM 

            #### Partie I : SVM Binaire ####

#I-1-
import pandas as pd
# Charger les données depuis le fichier CSV
Dataset = pd.read_csv("diabetes.csv")
# Afficher les premières lignes du jeu de données pour vérification
print(Dataset.head())


#I-2-
# Créer un nouveau vecteur Label à partir de la colonne 'Outcome'
Label = Dataset['Outcome'].copy()
# Afficher les premières valeurs du vecteur Label pour vérification
print(Label.head())


# #I-3-
# # Supprimer la colonne 'Outcome' du dataset
# Dataset.drop('Outcome', axis=1, inplace=True)
# # Afficher les premières lignes du dataset pour vérification
# print(Dataset.head())


#I-4-
from sklearn.model_selection import train_test_split
# Diviser le dataset et les labels en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(Dataset, Label, test_size=0.33, random_state=42)
# Afficher les tailles des ensembles d'apprentissage et de test pour vérification
print("Taille de l'ensemble d'apprentissage (X_train, y_train):", X_train.shape, y_train.shape)
print("Taille de l'ensemble de test (X_test, y_test):", X_test.shape, y_test.shape)


#I-5-
from sklearn.svm import SVC
# Créer une instance du modèle SVM binaire
svm_classifier = SVC(kernel='linear')  # Vous pouvez choisir le noyau approprié (linéaire, gaussien, polynomial, etc.)
# Entraîner le modèle sur les données d'apprentissage
svm_classifier.fit(X_train, y_train)
# Prédire les étiquettes pour les données de test
y_pred = svm_classifier.predict(X_test)


#I-6-
from sklearn.metrics import accuracy_score
# Calculer l'exactitude (accuracy) du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Exactitude (accuracy) du modèle:", accuracy)


#I-7-
 #a-
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.svm import SVC

# Définition des noyaux
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

# Initialisation des listes pour stocker les performances
accuracies = []
aucs = []
precisions = []
complexities = []

# Boucle sur chaque noyau
for kernel in kernels:
    # Création et entraînement du modèle SVM avec le noyau actuel
    svm_classifier = SVC(kernel=kernel)
    svm_classifier.fit(X_train, y_train)
    
    # Prédictions sur les données de test
    y_pred = svm_classifier.predict(X_test)
    
    # Calcul des métriques de performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred) if kernel != 'linear' else 'N/A'
    
    # Ajout des performances aux listes correspondantes
    accuracies.append(accuracy)
    precisions.append(precision)
    aucs.append(auc)
    
    # Estimation de la complexité de calcul (subjective)
    if kernel == 'linear':
        complexity = 'Low'
    elif kernel == 'rbf':
        complexity = 'Medium'
    else:
        complexity = 'High'
    complexities.append(complexity)

# Affichage des résultats dans un tableau
print("--------------------------------------------------------------------")
print("| Noyaux    |  Accuracy (ACC)  |  AUC      |  Précision  |  Complexité de calcul |")
print("--------------------------------------------------------------------")
for i in range(len(kernels)):
    print("| {:<9} | {:^18} | {:^9} | {:^12} | {:^22} |".format(kernels[i], round(accuracies[i], 4), aucs[i], round(precisions[i], 4), complexities[i]))
print("--------------------------------------------------------------------")


#b-
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Initialisation de la figure
plt.figure(figsize=(8, 6))

# Boucle sur chaque noyau
for kernel in kernels:
    # Création et entraînement du modèle SVM avec le noyau actuel
    svm_classifier = SVC(kernel=kernel, probability=True)
    svm_classifier.fit(X_train, y_train)
    
    # Prédictions sur les données de test
    y_pred_proba = svm_classifier.predict_proba(X_test)[:, 1]
    
    # Calcul de la courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # Tracé de la courbe ROC
    plt.plot(fpr, tpr, label=kernel)

# Ajout de la ligne diagonale pour la référence
plt.plot([0, 1], [0, 1], linestyle='--', color='black')

# Ajout de légendes et de titres
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbes ROC pour les différents noyaux de SVM')
plt.legend()
plt.grid(True)
plt.show()


#I-8-
import pandas as pd

# Supposons que Label contient les étiquettes de classe
Label = pd.Series([0, 1, 1, 0, 1, 0, 1, 0, 0, 1])  # Exemple de données fictives

# Afficher le nombre d'observations de chaque classe
print("Nombre d'observations par classe :")
print(Label.value_counts())


# I-9-
# Le nombre d'observations égal entre les classes (5 observations pour chaque classe) 
# indique un équilibre parfait entre les classes, ce qui peut faciliter la modélisation 
# et réduire le risque de biais dans les prédictions.


# I-10-
# Je recommande d'utiliser les SVM (Machines à Vecteurs de Support) pour la classification binaire
# car ils sont capables de traiter efficacement les ensembles de données linéaires et non linéaires,
# ce qui peut être avantageux lorsque les classes sont bien séparées.


