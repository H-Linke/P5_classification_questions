# script lançant la partie récuperation des datas
import pandas as pd

##########################
#        IMPORTS         #
##########################
from python_dev.preparation_data import *

##########################
#     CONFIGURATION      #
##########################
repertoire_courant = os.path.dirname(os.path.abspath(__file__))
# Remonter les répertoires parents jusqu'à atteindre le répertoire racine du projet
while not os.path.basename(repertoire_courant) == 'P5_classification_questions':
    repertoire_parent = os.path.dirname(repertoire_courant)
    if repertoire_parent == repertoire_courant:
        raise FileNotFoundError("Répertoire racine du projet introuvable.")
    repertoire_courant = repertoire_parent

path_data = os.path.join(repertoire_courant, 'gestion_data/ressources')

##########################
#       VARIABLES        #
##########################
n = input("Entrez le nombre minimal d'occurence pour un tag pour être conserver : ")

try:
    val = int(n)
    print("La valeur entree est un entier = ", val)
except ValueError:
    print("Ce n'est pas un entier!")


##########################
#       Fonctions        #
##########################


def main(data_path):
    """
    Cette fonction main récupère les données brut et les prépare pour
    être prête à être utilisée pour les modèles de machine learning

    :param data_path: commentaire sous forme de chaîne de caractères

    La fonction enregistre les données prêtes à l'emploi
    """

    data_brut_path = os.path.join(data_path, 'brut')
    data_trans_path = os.path.join(data_path, 'transformees/Questions.csv')
    tags_cons_path = os.path.join(data_path, 'transformees/Tags_cons.csv')

    # Recuperation des données d'intérêt uniquement
    data = step_0_formalisation(data_brut_path)

    data.Question = step_1_3_main_normalisation_texte(data.Question)

    data.Tags, tags_cons = step_2_preparation_tags(data.Tags, nbr_occ=val)

    data.to_csv(data_trans_path, index=False)

    data = pd.read_csv(data_trans_path)

    print('shape1 :', data.shape)
    print("nbr na", data.Tags.isna().sum())

    data.dropna(how='any', ignore_index=True, inplace=True)

    print('shape2 :', data.shape)

    data.to_csv(data_trans_path, index=False)
    tags_cons.to_csv(tags_cons_path, index=False)


if __name__ == '__main__':
    main(path_data)
