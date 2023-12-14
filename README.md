# Catégorisez automatiquement des questions

Le but de ce projet est de proposer une ia permettant de suggérer automatiquement des tags en lien aux questions posées 
par les utilisateurs de __Stack Overflow__

## Décomposition du projet

Dans le dossier projet ce trouve le notebook qui regroupe toutes les premières analyses, ainsi que les modèles non 
supervisés et une ébauche des modèles supervisés qui a servi de base pour un développement MLOps de ces derniers.

Pour le reste, le projet est divisé en 3 parties qui correspondent à 3 dossiers :
1. Le dossier `gestion_data` qui regroupe traite et sélectionne toutes les datas.
Ici sont réalisés les traitements préliminaires, à savoir la normalisation des données.
2. Le dossier `gestion_modeles` qui gère tout ce qui a lieu à la création des modèles. 3 étapes sont effectuées 
séparément : 
    * Transformation des questions en vecteurs (embedding), selon différentes méthodes. Les résultats sont enregistrés
au format csv séparés en fichier train et tests ainsi que les Tags sous forme encodée.
    * Partie MLOps, des modèles sont créés entraîner et évaluer en série de manière à pouvoir sélectionner le plus 
performant. Ceci est réaliser avec un suivi __MLFLOW__.
    * Le modèle le plus performant est entraîné et enregistré dans MLFLOW avec le maximum de données.
3. Le dossier `production` dans lequel est développé le script permettant de mettre en production sur __Streamlite__ le 
meilleur modèle.

Toutes ces différentes sont paramétrables et exécutés via les fichiers `__main__.py`. Faute de plateforme, toutes les 
résultats générés par les différents scripts, sont enregistrés dans des dossiers au même endroit que le lanceur 
__main__.py