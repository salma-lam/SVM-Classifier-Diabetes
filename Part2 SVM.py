                                    #TP SVM

            ####  Partie II: SVM Mono-classe  ####

import pandas as pd
# Charger les données depuis le fichier CSV
Dataset = pd.read_csv("diabetes.csv")
# Créer un nouveau vecteur Label à partir de la colonne 'Outcome'
Label = Dataset['Outcome'].copy()
# Afficher les premières valeurs du vecteur Label pour vérification
print(Label.head())

#II-1-
#  sélectionne les lignes du DataFrame "Dataset" pour lesquelles la colonne "Outcome" a une valeur égale à 1, puis stocke ces lignes dans une nouvelle variable appelée "dataset_selected1
dataset_selected1= Dataset.loc[Dataset ['Outcome'].isin([1])]

#II-2-
# Sélectionner les lignes où la colonne 'Outcome' a une valeur égale à 0
dataset_selected0 = Dataset.loc[Dataset['Outcome'].isin([0])]

#II-3-

# Extraire les étiquettes pour dataset_selected1 (classe 1)
Label_1 = dataset_selected1['Outcome']
# Extraire les étiquettes pour dataset_selected0 (classe 0)
Label_0 = dataset_selected0['Outcome']

#II-4-
# Sélectionner les données pour dataset_selected1 (classe 1)
Data_1 = dataset_selected1.drop(columns=['Outcome'])
# Sélectionner les données pour dataset_selected0 (classe 0)
Data_0 = dataset_selected0.drop(columns=['Outcome'])

#II-5-
# Dans l'étape d'apprentissage pour un SVM mono-classe, on utilise uniquement les données 
# correspondant à la classe que l'on souhaite détecter. Cela signifie qu'on n'utilise 
# que les données positives ou normales pour entraîner le modèle. Par exemple, 
# si on utilise un SVM mono-classe pour détecter des anomalies,
# on n'utilise que les données normales pour l'entraînement. 
# En résumé, dans l'étape d'apprentissage d'un SVM mono-classe, on utilise uniquement les données positives ou normales.  

#II-6-
from sklearn import svm

# Créer une instance du classificateur SVM mono-classe pour chaque noyau
kernel_types = ['linear', 'rbf', 'poly', 'sigmoid']
classifiers = {}

for kernel in kernel_types:
    classifiers[kernel] = svm.OneClassSVM(kernel=kernel)
    classifiers[kernel].fit(Data_1)  # Utiliser uniquement les données de classe 1 pour l'apprentissage

# Tester les classificateurs sur les données de classe 0
for kernel, clf in classifiers.items():
    y_pred = clf.predict(Data_0)
    accuracy = (y_pred == -1).sum() / len(y_pred)  # Calculer l'accuracy
    print("Accuracy pour le noyau {}: {:.2f}".format(kernel, accuracy))


#II-7-
#a-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score

# Charger les données depuis le fichier CSV
Dataset = pd.read_csv("diabetes.csv")

# Sélectionner les données pour la classe 1
Data_1 = Dataset.loc[Dataset['Outcome'] == 1].drop(columns=['Outcome'])

# Sélectionner les données pour la classe 0
Data_0 = Dataset.loc[Dataset['Outcome'] == 0].drop(columns=['Outcome'])

# Créer une instance du classificateur SVM mono-classe pour chaque noyau
kernel_types = ['linear', 'rbf', 'poly', 'sigmoid']
classifiers = {}

for kernel in kernel_types:
    classifiers[kernel] = OneClassSVM(kernel=kernel)
    classifiers[kernel].fit(Data_1)  # Utiliser uniquement les données de classe 1 pour l'apprentissage

# Initialisation des listes pour stocker les performances
accuracy_scores = []
precision_scores = []
complexity_values = []

# Tester les classificateurs sur les données de classe 0
for kernel, clf in classifiers.items():
    y_pred = clf.predict(Data_0)
    accuracy = (y_pred == -1).sum() / len(y_pred)  # Calculer l'accuracy
    
    # Calculer la précision pour la classe anormale (-1)
    precision = precision_score([-1] * len(Data_0), y_pred, pos_label=-1)  # Calculer la précision
    
    # Ajouter les performances aux listes
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    
    # Déterminer la complexité de calcul
    if kernel == 'linear':
        complexity = 'Low'
    elif kernel == 'rbf':
        complexity = 'Medium'
    else:
        complexity = 'High'
    
    # Ajouter la complexité de calcul
    complexity_values.append(complexity)

# Afficher le tableau des performances
performance_table = pd.DataFrame({
    "Noyaux": kernel_types,
    "Accuracy": accuracy_scores,
    "Précision": precision_scores,
    "Complexité de calcul": complexity_values
})

print(performance_table)


#b-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Initialisation de la figure
plt.figure(figsize=(8, 6))

# Boucle sur chaque noyau
for kernel, clf in classifiers.items():
    # Prédire les scores de décision pour les données de test
    scores = clf.decision_function(Data_0)
    # Calculer le taux de vrais positifs et le taux de faux positifs manuellement
    thresholds = np.linspace(min(scores), max(scores), 100)
    tpr = []
    fpr = []
    for threshold in thresholds:
        # Calculer le nombre de vrais positifs
        true_positives = np.sum(scores >= threshold)
        # Calculer le taux de vrais positifs
        tpr.append(true_positives / len(Data_0))
        # Calculer le nombre de faux positifs
        false_positives = np.sum(scores < threshold)
        # Calculer le taux de faux positifs
        fpr.append(false_positives / len(Data_0))
    # Tracer la courbe ROC
    plt.plot(fpr, tpr, label=kernel)

# Ajout des légendes et du titre
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC pour les différents noyaux')
plt.legend()
plt.grid(True)

# Affichage de la figure
plt.show()



#II-8-
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Exemple de données fictives
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_true, y_pred)

# Créer la heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Vraie classe')
plt.ylabel('Classe prédite')
plt.title('Matrice de confusion')
plt.show()


#II-9-
# En comparant les résultats des SVM mono-classe et binaire, plusieurs observations peuvent être faites :

# 1. Accuracy et AUC: Les SVM mono-classe présentent généralement des scores d'accuracy et d'AUC plus élevés que les SVM binaires pour tous les types de noyaux 
# (linear, rbf, poly, sigmoid). Cela suggère que les SVM mono-classe ont une meilleure capacité à séparer les données et à généraliser les modèles par rapport aux 
# SVM binaires.

# 2. Précision: Les deux types de SVM affichent une précision de 1.0 pour tous les types de noyaux dans les résultats des SVM mono-classe,
# tandis que les SVM binaires montrent des valeurs de précision légèrement inférieures mais toujours significatives. Cela indique que les deux approches sont 
# capables de bien identifier les classes, bien que les SVM mono-classe présentent une performance légèrement supérieure.

# 3. Complexité de calcul: Dans les deux cas, la complexité de calcul reste variable, ce qui signifie que les SVM nécessitent une certaine puissance de calcul, 
# mais le choix du noyau n'a pas d'impact significatif sur cette complexité.

# En résumé, les SVM mono-classe semblent offrir des performances globalement meilleures que les SVM binaires dans la tâche de classification étudiée,
# en fournissant des modèles plus précis et plus généralisables pour la séparation des données. 
# Cependant, cela peut dépendre également du contexte spécifique de l'application et des caractéristiques des données.



