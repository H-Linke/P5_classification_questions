# Scrip principal concernant le lancement de la partie embedding des questions

##########################
#        IMPORTS         #
##########################
from gestion_modeles.fonctions.models import *

##########################
#     CONFIGURATION      #
##########################
repertoire_projet = os.path.dirname(os.path.abspath(__file__))
# Remonter les répertoires parents jusqu'à atteindre le répertoire racine du projet
while not os.path.basename(repertoire_projet) == 'P5_classification_questions':
    repertoire_parent = os.path.dirname(repertoire_projet)
    if repertoire_parent == repertoire_projet:
        raise FileNotFoundError("Répertoire racine du projet introuvable.")
    repertoire_projet = repertoire_parent

##########################
#      Variables         #
##########################

donnees_source = os.path.join(
    repertoire_projet, 'gestion_data/ressources/transformees/Questions.csv')


##########################
#       Fonction         #
##########################
def main(file_path):
    """
    Cette fonction charge le fichier csv et transforme les données :
        embedding pour les questions et one hot encoding pour les Tags
        A la fin, elle les enregistre au format csv dans le dossier ressource
    :param file_path: path du fichier csv en entrée
    """

    questions_serie, tags_dummies = step_util_load_data(file_path)
    X_train, X_test, Y_train, Y_test = step_util_sep_data(questions_serie, tags_dummies)
    # step_0_1_doc2vec(X_train, X_test, Y_train, Y_test)
    step_0_2_bert_embedding(questions_serie, tags_dummies, pretraitement=False)
    step_0_3_use_embedding(questions_serie, tags_dummies, pretraitement=False)


if __name__ == '__main__':
    main(donnees_source)
    print('Les données ont été correctement enregistrées')
