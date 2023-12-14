# Scrip principal lançant toute la partie entraînement des modèles

##########################
#        IMPORTS         #
##########################
import itertools

import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

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
#       Fonction         #
##########################


def main():
    """
    Cette fonction lance l'entraînement des différents modèles en faisant varier les paramètres
    """

    use_path_2 = f"{repertoire_projet}/gestion_modeles/ressources_embedding/use/use_False"

    param_train_models = {
        'embedding_path': [use_path_2],
        'model_boite': [step_2_2_multi_output],
        'model_base': [step_3_1_regression_logistique]
    }

    # Obtention de toutes les combinaisons possibles de paramètres
    param_combinations = list(itertools.product(*param_train_models.values()))

    for embedding_path, model_boite, model_base in param_combinations:

        X_train = pd.read_csv(f"{embedding_path}_X_train.csv", index_col=0)
        Y_train = pd.read_csv(f"{embedding_path}_Y_train.csv", index_col=0)

        X_test = pd.read_csv(f"{embedding_path}_X_test.csv", index_col=0)
        Y_test = pd.read_csv(f"{embedding_path}_Y_test.csv", index_col=0)

        # Initialisation du modèle
        model = model_boite(model_base())

        print(f'Debut modèle : {embedding_path.split("/")[-2]}, {model_boite.__name__[9:]}, '
              f'{model_base.__name__[9:]}')

        jaccard, f1, f2, f3, hamming = step_4_entrainement_et_performances(model, X_train, Y_train, X_test, Y_test)

        mlflow.set_experiment("Modèle abouti")

        with mlflow.start_run(
                run_name=f'{embedding_path.split("/")[-1]}_{model_boite.__name__[9:]}_{model_base.__name__[9:]}'
        ):
            # Enregistrement du modèle
            mlflow.sklearn.log_model(model, 'Best_Model')

            # Log concernant les caractéristiques du modèle entraîné
            mlflow.log_param('Vectoriseur', embedding_path.split("/")[-1])
            mlflow.log_param('model_boite', model_boite.__name__[9:])
            mlflow.log_param('model_base', model_base.__name__[9:])

            # Log des moyennes des scores sur les folds
            mlflow.log_metric('jaccard', jaccard)
            mlflow.log_metric('f1', f1)
            mlflow.log_metric('f2', f2)
            mlflow.log_metric('f3', f3)
            mlflow.log_metric('hamming', hamming)


if __name__ == '__main__':
    main()
