# Ce script contient toutes les fonctions liées à la récupération et au prétraitement des données

##########################
#        IMPORTS         #
##########################
import nltk

nltk.download('punkt')
nltk.download("wordnet")
nltk.download('omw-1.4')

import os
import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import streamlit as st
import mlflow

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from utils.constantes import emoticons_regex, stop_words, Tags

##########################
#     CONFIGURATION      #
##########################
repertoire_projet = os.getcwd().replace(r"\\", "/")


@st.cache_resource
def replace_characters(text):
    text = re.sub(emoticons_regex, " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text


@st.cache_resource
def step_1_1_clean_text(comment):
    """
    Cette fonction prend un commentaire en paramètre et le renvoie normalisé
    :param comment: commentaire sous forme de chaîne de caractères
    :return: commentaire normalisé
    """
    # On met tout en minuscules
    comment = comment.lower()

    # On supprime les formes contractées du texte pour uniformiser le format du texte avec la forme décontractée
    comment = contractions.fix(comment)

    # Remplacement de certaines parties du texte telles que les utilisateurs, les liens, les mentions
    comment = re.sub(r'https?://\S+', ' ', comment)
    comment = re.sub(r'[^\w$@#_+<>]', ' ', comment)
    comment = re.sub(r'\s+', ' ', comment)

    return comment


@st.cache_resource
def step_1_2_lemmatize_text(comment):
    """
    Cette fonction lemmatise le commente
    :param comment: commentaire
    :return: commentaire lemmatisé
    """
    # Lemmatise
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in comment.split()]
    lemmatized_comment = ' '.join(lemmatized_words)

    return lemmatized_comment


@st.cache_resource
def step_1_3_main_normalisation_texte(data):
    """
    Cette fonction prend en entrée une Serie pandas dont elle normalise le texte
    Un DataFrame nettoyé et normalisé.

    :param data: Serie pandas contenant du texte
    :return: Serie pandas avec le texte normalisé
    """

    data = data.apply(replace_characters)\
        .apply(step_1_1_clean_text) \
        .apply(step_1_2_lemmatize_text) \
        .apply(word_tokenize) \
        .apply(lambda comm: [word for word in comm if word not in stop_words]) \
        .apply(lambda x: ' '.join(x))

    return data


@st.cache_resource
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


@st.cache_resource
def chargement_models_predict():

    # Chargement du modèle de prediction
    model_path = f"{repertoire_projet}/gestion_modeles/mlruns/234254233934860191/64b3b462e5224309acf18f0cff031c60/artifacts/Best_Model_ACP_std"
    predic_model = mlflow.sklearn.load_model(model_path)

    return predic_model

@st.cache_resource
def chargement_scaler():

    # Chargement du modèle de prediction
    model_path = f"{repertoire_projet}/gestion_modeles/mlruns/234254233934860191/64b3b462e5224309acf18f0cff031c60/artifacts/scaler"
    scaler = mlflow.sklearn.load_model(model_path)

    return scaler

@st.cache_resource
def chargement_pca():

    # Chargement du modèle de prediction
    model_path = f"{repertoire_projet}/gestion_modeles/mlruns/234254233934860191/64b3b462e5224309acf18f0cff031c60/artifacts/ACP"
    ACP = mlflow.sklearn.load_model(model_path)

    return ACP


@st.cache_resource
def use_and_predict(data, _emb_model, _scaler_model, _acp_model, _pred_model):
    """

    :param data: données à prédire sous forme de liste, Serie
    :return: prediction des Tags
    """

    data_vec = _emb_model(data).numpy()
    data_vec_std = _scaler_model.transform(data_vec)
    data_vec_acp = _acp_model.transform(data_vec_std)
    data_pred = pd.Series(_pred_model.predict(data_vec_acp)[0], index=Tags)

    pred = data_pred[data_pred == 1].index

    return list(pred)


