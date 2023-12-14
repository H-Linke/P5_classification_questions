# Ce script contient toutes les fonctions liées à la mise en place des modèles

##########################
#        IMPORTS         #
##########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import jaccard_score, f1_score, fbeta_score, hamming_loss, accuracy_score, make_scorer

import mlflow
import mlflow.pyfunc

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
from gensim.utils import simple_preprocess

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

# Get the current date and time
now = datetime.now()

# Format the date and time as AAAAMMJJhhmmss
formatted_date = now.strftime("%Y%m%d%H%M%S")


# path_image = os.path.join(repertoire_projet, config.get('PATH_MODELS', 'path_image'))


######################
#       Utils        #
######################

def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations


def pretraitement_text(questions):

    dt_corpus = questions.apply(lambda x: x.split()).values.tolist()
    data_words = list(sent_to_words(dt_corpus))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    # Form Bigrams
    data_words_bigrams = pd.Series([bigram_mod[doc] for doc in data_words]).apply(lambda mots: ' '.join(mots))

    return data_words_bigrams


def bert_embedding_model():
    # Charger le modèle BERT
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
                                     name="bert_preprocess")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", name="bert_encoder")

    # Entrée du texte
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

    # Prétraitement BERT
    preprocessed_text = bert_preprocess(text_input)

    # Sortie de l'encodeur BERT (pooled_output)
    pooled_output = bert_encoder(preprocessed_text)['pooled_output']

    # Créer le modèle
    model = tf.keras.Model(inputs=text_input, outputs=pooled_output)

    return model


def use_embedding_model():
    # Chargement du modèle Universal Sentence Encoder
    use_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(use_url)

    # Fonction d'embedding qui prend une série pandas en entrée
    def embed_text_series(text_series):

        # Calcul de l'embedding pour chaque texte
        embeddings = embed(text_series)

        return embeddings

    return embed_text_series


##########################
#       Fonctions        #
##########################

# Chargement du fichier de data

def step_util_load_data(file_path):
    """
    Cette fonction charge un fichier CSV dans une série pandas
    :param file_path: fichier CSV
    :return: Série pandas
    """
    data = pd.read_csv(file_path)
    questions_serie = data.Question

    tags_split = data.Tags.str.extractall(r'(\S+)')
    tags_dummies = pd.get_dummies(tags_split[0], prefix='Tag').groupby(level=0).max().astype('int')

    return questions_serie, tags_dummies


def step_util_sep_data(questions_serie, tags_dummies):
    X_train, X_test, Y_train, Y_test = train_test_split(questions_serie, tags_dummies)
    return X_train, X_test, Y_train, Y_test


# Méthodes d'embedding
def step_0_1_doc2vec(X_train_text, X_test_text, Y_train, Y_test, dm=1, vect_size=100, min_count=5, neg=5, ns_exponent=0.75, seed=0):
    """
    Méthode : Doc2Vec
    Cette fonction prend un Serie pandas en entrée contenant
    du texte et le convertit en vecteur en utilisant
    la méthode d'embedding Doc2Vec.

    :param X_train_text: Serie pandas contenant du texte à convertir en vecteurs
    :param X_test_text: Serie pandas contenant du texte à convertir en vecteurs
    :param X_train: DataFrame pandas contenant du texte à convertir en vecteurs
    :param X_test: DataFrame pandas contenant du texte à convertir en vecteurs
    :param dm: paramètre de Doc2Vec (0 pour DBOW, 1 pour DM)
    :param vect_size: dimension des vecteurs de document
    :param min_count: nombre minimum d'occurrences d'un mot pour qu'il soit pris en compte
    :param neg: nombre de "mots de bruit" échantillonnés lors de l'apprentissage
    :param ns_exponent: exposant utilisé pour façonner la distribution de l'échantillonnage négatif
    :param seed: graine aléatoire pour la reproductibilité des résultats
    :return: DataFrame contenant les vecteurs correspondant à chaque commentaire
    """

    # Definition du nom pour sauvegarde
    path_to_save = \
        f"gestion_modeles/ressources_embedding/doc2vec/doc2v_{dm}_{vect_size}_{min_count}_{neg}_{ns_exponent}_{seed}"
    name_xtrain = os.path.join(repertoire_projet, f"{path_to_save}_X_train.csv")
    name_xtest = os.path.join(repertoire_projet, f"{path_to_save}_X_test.csv")
    name_ytrain = os.path.join(repertoire_projet, f"{path_to_save}_Y_train.csv")
    name_ytest = os.path.join(repertoire_projet, f"{path_to_save}_Y_test.csv")

    # Création d'une liste de TaggedDocuments
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train_text)]

    # Entraînement d'un modèle Doc2Vec
    model = Doc2Vec(
        documents, vector_size=vect_size, dm=dm, window=5,
        min_count=min_count, epochs=20, seed=seed, workers=8, negative=neg,
        ns_exponent=ns_exponent, dm_mean=1
    )

    # Chemin où vous souhaitez enregistrer le modèle Gensim
    name = f'Doc2Vec_model_{vect_size}_{dm}_{min_count}_{neg}_{ns_exponent}_{seed}_{formatted_date}'
    gensim_model_path = os.path.join(repertoire_projet, "gestion_modeles/models/gensim_models", name)

    # Enregistrement du modèle Gensim
    model.save(gensim_model_path)

    # Enregistrement des vecteurs de chaque commentaire
    X_train_vec = X_train_text.apply(lambda words: model.infer_vector(words))
    X_train_vec = pd.DataFrame(list(X_train_vec.apply(list)))

    X_test_vec = X_test_text.apply(lambda words: model.infer_vector(words))
    X_test_vec = pd.DataFrame(list(X_test_vec.apply(list)))

    X_train_vec.to_csv(name_xtrain)
    X_test_vec.to_csv(name_xtest)
    Y_train.to_csv(name_ytrain)
    Y_test.to_csv(name_ytest)


def step_0_2_bert_embedding(questions_serie, tags_dummies, pretraitement=True):
    """
    Cette fonction effectue le word embedding des questions avec la méthode bert.
    Puis la fonction enregistre au format csv toutes les données près pour entraînement des modèles :
    un X_train, X_test, Y_train, Y_test

    :param questions_serie: Serie Pandas contenant les questions
    :param tags_dummies: DataFrame contenant le one hot encoding des Tags
    :param pretraitement: effectue ou nom un prétraitement permettant de selectionner
    les mots les plus connus dans les questions. Cette fonction sert à réduire le nombre de mots dans une question
    étant donné que le modèle bert ne prend en compte que les 128 premiers mots
    """

    if pretraitement:
        questions = pretraitement_text(questions_serie)
    else:
        questions = questions_serie.copy()

    # Initialisation du modèle bert
    bert = bert_embedding_model()

    # Embedding des questions
    transformed_data = pd.DataFrame(bert.predict(questions))

    X_train, X_test, Y_train, Y_test = step_util_sep_data(transformed_data, tags_dummies)

    # Definition du nom pour sauvegarde
    path_to_save = \
        f"gestion_modeles/ressources_embedding/bert/bert_{pretraitement}"
    name_xtrain = os.path.join(repertoire_projet, f"{path_to_save}_X_train.csv")
    name_xtest = os.path.join(repertoire_projet, f"{path_to_save}_X_test.csv")
    name_ytrain = os.path.join(repertoire_projet, f"{path_to_save}_Y_train.csv")
    name_ytest = os.path.join(repertoire_projet, f"{path_to_save}_Y_test.csv")

    X_train.to_csv(name_xtrain)
    X_test.to_csv(name_xtest)
    Y_train.to_csv(name_ytrain)
    Y_test.to_csv(name_ytest)


def step_0_3_use_embedding(questions_serie, tags_dummies, pretraitement=True, batch_size=1000):
    """
    Cette fonction effectue le word embedding des questions avec la méthode use.
    Puis la fonction enregistre au format csv toutes les données près pour entraînement des modèles :
    un X_train, X_test, Y_train, Y_test

    :param questions_serie: Serie Pandas contenant les questions
    :param tags_dummies: DataFrame contenant le one hot encoding des Tags
    :param pretraitement: effectue ou nom un prétraitement permettant de selectionner
    les mots les plus connus dans les questions. Cette fonction sert à réduire le nombre de mots dans une question
    étant donné que le modèle use ne prend en compte que les 128 premiers mots
    """

    if pretraitement:
        questions = pretraitement_text(questions_serie)
    else:
        questions = questions_serie

    # Initialisation du modèle use
    use = use_embedding_model()

    embeddings_list = []
    for i in range(0, len(questions), batch_size):
        batch_texts = questions[i:i + batch_size]
        embeddings_list.append(use(batch_texts).numpy())

    # Concaténation des embeddings de tous les lots
    embeddings = np.concatenate(embeddings_list, axis=0)

    # Embedding des questions
    transformed_data = pd.DataFrame(list(embeddings))

    X_train, X_test, Y_train, Y_test = step_util_sep_data(transformed_data, tags_dummies)

    # Definition du nom pour sauvegarde
    path_to_save = \
        f"gestion_modeles/ressources_embedding/use/use_{pretraitement}"
    name_xtrain = os.path.join(repertoire_projet, f"{path_to_save}_X_train.csv")
    name_xtest = os.path.join(repertoire_projet, f"{path_to_save}_X_test.csv")
    name_ytrain = os.path.join(repertoire_projet, f"{path_to_save}_Y_train.csv")
    name_ytest = os.path.join(repertoire_projet, f"{path_to_save}_Y_test.csv")

    X_train.to_csv(name_xtrain)
    X_test.to_csv(name_xtest)
    Y_train.to_csv(name_ytrain)
    Y_test.to_csv(name_ytest)


##################################
#      MODELES ASSEMBLEUR        #
##################################


def step_2_1_one_vs_rest(model_base):
    ovr = OneVsRestClassifier(model_base)
    return ovr


def step_2_2_multi_output(model_base):
    ovr = MultiOutputClassifier(model_base, n_jobs=-1)
    return ovr


###############################
#      MODELES DE BASE        #
###############################


def step_3_1_regression_logistique():
    """
    Modèle de régression logistique

    :return: modèle
    """

    clf = LogisticRegression(class_weight='balanced', solver='newton-cholesky')
    return clf


def step_3_2_sgdc():
    """
    Modèle SGDClassifier

    :return: modèle
    """
    sgdc = SGDClassifier(class_weight='balanced', loss='log_loss', random_state=0)
    return sgdc


###########################################
#     ENTRAÎNEMENT ET PERFORMANCES        #
###########################################


def step_4_scoring():
    # Définir les métriques personnalisées avec make_scorer
    jaccard_scorer = make_scorer(jaccard_score, greater_is_better=True, average='samples')
    f1_scorer = make_scorer(f1_score, greater_is_better=True, average='samples')
    f2_scorer = make_scorer(fbeta_score, greater_is_better=True, beta=2, average='samples')
    f3_scorer = make_scorer(fbeta_score, greater_is_better=True, beta=3, average='samples')
    hamming_scorer = make_scorer(hamming_loss, greater_is_better=False)

    # scoring avec les métriques personnalisées
    scoring = {
        'jaccard': jaccard_scorer,
        'f1': f1_scorer,
        'f2': f2_scorer,
        'f3': f3_scorer,
        'hamming': hamming_scorer
    }

    return scoring


def step_4_entrainement_et_performances(entire_model, X_train, Y_train, X_test, Y_test):
    entire_model.fit(X_train, Y_train)

    # Prédictions faîtes par le modèle
    Y_pred = entire_model.predict(X_test)

    jaccard = jaccard_score(Y_test, Y_pred, average='samples')
    f1 = f1_score(Y_test, Y_pred, average='samples')
    f2 = fbeta_score(Y_test, Y_pred, beta=2, average='samples')
    f3 = fbeta_score(Y_test, Y_pred, beta=3, average='samples')
    hamming = hamming_loss(Y_test, Y_pred)

    return jaccard, f1, f2, f3, hamming
