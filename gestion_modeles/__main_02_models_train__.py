# Scrip principal lançant toute la partie entraînement des modèles

##########################
#        IMPORTS         #
##########################
import itertools

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

    echantillon = 15000

    # Créer un DataFrame pour stocker les informations
    mlflow_df_columns = [
        'Vectoriseur', 'model_boite', 'model_base',
        'mean_jaccard_score', 'mean_f1_score', 'mean_f2_score', 'mean_f3_score', 'mean_hamming_score',
        'std_jaccard_score', 'std_f1_score', 'std_f2_score', 'std_f3_score', 'std_hamming_score'
    ]

    mlflow_df = pd.DataFrame(columns=mlflow_df_columns)

    doc2v_path = f"{repertoire_projet}/gestion_modeles/ressources_embedding/doc2vec/doc2v_1_100_5_5_0.75_0"
    bert_path_1 = f"{repertoire_projet}/gestion_modeles/ressources_embedding/bert/bert_True"
    use_path_1 = f"{repertoire_projet}/gestion_modeles/ressources_embedding/use/use_True"
    bert_path_2 = f"{repertoire_projet}/gestion_modeles/ressources_embedding/bert/bert_False"
    use_path_2 = f"{repertoire_projet}/gestion_modeles/ressources_embedding/use/use_False"

    param_train_models = {
        'embedding_path': [doc2v_path, bert_path_1, use_path_1, bert_path_2, use_path_2],
        'model_boite': [step_2_1_one_vs_rest, step_2_2_multi_output],
        'model_base': [step_3_1_regression_logistique, step_3_2_sgdc]
    }

    # Obtention de toutes les combinaisons possibles de paramètres
    param_combinations = list(itertools.product(*param_train_models.values()))

    for embedding_path, model_boite, model_base in param_combinations:

        X_train = pd.read_csv(f"{embedding_path}_X_train.csv", index_col=0)\
            .reset_index(drop=True).sample(n=echantillon, random_state=0)
        Y_train = pd.read_csv(f"{embedding_path}_Y_train.csv", index_col=0)\
            .reset_index(drop=True).sample(n=echantillon, random_state=0)

        # X_test = pd.read_csv(f"{embedding_path}_X_test.csv")
        # Y_test = pd.read_csv(f"{embedding_path}_Y_test.csv", index_col=0)

        # Initialisation du modèle
        model = model_boite(model_base())

        # Créer le scoring avec greater_is_better=True
        scoring = step_4_scoring()

        print(f'Debut modèle : {embedding_path.split("/")[-2]}, {model_boite.__name__[9:]}, '
              f'{model_base.__name__[9:]}')

        # Effectuer la validation croisée avec le scoring
        scores = cross_validate(model, X_train, Y_train, scoring=scoring, cv=3)

        print('Fin')

        with mlflow.start_run(
                run_name=f'{embedding_path.split("/")[-1]}_{model_boite.__name__[9:]}_{model_base.__name__[9:]}'
        ):
            # Log concernant les caractéristiques du modèle entraîné
            mlflow.log_param('Vectoriseur', embedding_path.split("/")[-1])
            mlflow.log_param('model_boite', model_boite.__name__[9:])
            mlflow.log_param('model_base', model_base.__name__[9:])

            # Log des moyennes des scores sur les folds
            mlflow.log_metric('mean_jaccard_score', scores['test_jaccard'].mean())
            mlflow.log_metric('mean_f1_score', scores['test_f1'].mean())
            mlflow.log_metric('mean_f2_score', scores['test_f2'].mean())
            mlflow.log_metric('mean_f3_score', scores['test_f3'].mean())
            mlflow.log_metric('mean_hamming_score', scores['test_hamming'].mean())

            # Log des écarts-types des scores sur les folds
            mlflow.log_metric('std_jaccard_score', scores['test_jaccard'].std())
            mlflow.log_metric('std_f1_score', scores['test_f1'].std())
            mlflow.log_metric('std_f2_score', scores['test_f2'].std())
            mlflow.log_metric('std_f3_score', scores['test_f3'].std())
            mlflow.log_metric('std_hamming_score', scores['test_hamming'].std())

            mlflow.log_metric('time', scores['fit_time'].mean())

            # Ajouter les informations dans le DataFrame
            mlflow_df = pd.concat([mlflow_df, pd.DataFrame({
                'Vectoriseur': [embedding_path.split("/")[-1]],
                'model_boite': [model_boite.__name__[9:]],
                'model_base': [model_base.__name__[9:]],
                'mean_jaccard_score': [scores['test_jaccard'].mean()],
                'mean_f1_score': [scores['test_f1'].mean()],
                'mean_f2_score': [scores['test_f2'].mean()],
                'mean_f3_score': [scores['test_f3'].mean()],
                'mean_hamming_score': [scores['test_hamming'].mean()],
                'std_jaccard_score': [scores['test_jaccard'].std()],
                'std_f1_score': [scores['test_f1'].std()],
                'std_f2_score': [scores['test_f2'].std()],
                'std_f3_score': [scores['test_f3'].std()],
                'std_hamming_score': [scores['test_hamming'].std()],
            })], ignore_index=True)
    # Enregistrez le DataFrame au format CSV
    mlflow_df.to_csv(f'{repertoire_projet}/gestion_modeles/results/mlflow_results.csv', index=False)


if __name__ == '__main__':
    main()
