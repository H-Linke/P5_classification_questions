# Ce script contient toutes les fonctions liées à la récupération et au prétraitement des données

##########################
#        IMPORTS         #
##########################
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

# from constantes import emoticons_regex, stop_words, Tags


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
#       Constantes        #
##########################

emoticons_regex = r"(^|\s)(:\)|[\d\s]+|I|:\(|;\)|:D|;D|:p|:P|:o|:O|:@|:s|:S|:\$|:\||:\\|:\/|:\'\\(|:\'\\)|:\*|<3)($|\s)"

stop_words = {
    'doesn', 'both', 'ourselves', 'at', "hasn't", 'you', 'will', "shouldn't", 'were', 'am', 'ours', 'won', 'me',
    'below', 'she', 'not', 'did', 'that', 'it', 'wouldn', 'as', "you're", "should've", 'my', "isn't", "mightn't",
    "you'd", 'be', "it's", 't', 'he', "you've", "didn't", 'him', 'so', 'under', 'any', 've', 'down', 'most', 'over',
    "won't", "that'll", 'once', 'had', 'was', "wasn't", 'if', 'of', 'isn', "wouldn't", 'herself', 'o', 'are', 'no',
    'just', 'theirs', 'a', 'nor', 'mustn', 'needn', 'itself', 'aren', "needn't", 'its', 'being', 'up', 'been', 'to',
    'but', 'do', 'couldn', 'ain', 'yourself', 'or', 'than', "mustn't", 'too', 'her', 's', 're', "you'll", 'the',
    "couldn't", 'until', 'by', "shan't", 'them', 'your', 'y', 'some', 'such', 'few', 'more', 'didn', 'out', 'through',
    'myself', 'haven', 'himself', 'don', 'm', 'mightn', 'above', 'doing', 'shouldn', 'we', 'should', 'his', 'off',
    "haven't", 'shan', 'because', 'whom', 'having', 'and', "she's", 'wasn', 'd', 'own', "aren't", 'then', 'has',
    'these', "doesn't", 'themselves', 'weren', 'have', 'here', 'hasn', 'against', 'between', "weren't", 'yourselves',
    'an', 'does', 'll', 'on', 'yours', 'from', 'our', 'there', "don't", 'hers', 'during', 'into', 'ma', "hadn't",
    'hadn', 'those', 'this', 'i', 'is', 'for', 'in', 'with', 'can'
}


Tags = ['.net', '.net-core', 'ajax', 'algorithm', 'amazon-web-services', 'android', 'android-layout', 'android-studio',
        'angular', 'angularjs', 'apache', 'apache-spark', 'api', 'arrays', 'asp.net', 'asp.net-core', 'asp.net-mvc',
        'asp.net-web-api', 'assembly', 'async-await', 'asynchronous', 'authentication', 'bash', 'c', 'c#', 'c++',
        'c++11', 'class', 'cocoa', 'cocoa-touch', 'css', 'database', 'dataframe', 'date', 'datetime', 'debugging',
        'dictionary', 'django', 'docker', 'eclipse', 'entity-framework', 'exception', 'express', 'file', 'firebase',
        'flutter', 'forms', 'function', 'gcc', 'generics', 'git', 'google-chrome', 'gradle', 'haskell', 'hibernate',
        'html', 'http', 'image', 'ios', 'ipad', 'iphone', 'java', 'java-8', 'javascript', 'jpa', 'jquery', 'json',
        'kotlin', 'lambda', 'language-lawyer', 'laravel', 'linq', 'linux', 'list', 'logging', 'machine-learning',
        'macos', 'math', 'matplotlib', 'maven', 'memory', 'mongodb', 'multithreading', 'mysql', 'node.js', 'npm',
        'numpy', 'objective-c', 'oop', 'opencv', 'optimization', 'pandas', 'performance', 'php', 'postgresql', 'python',
        'python-2.7', 'python-3.x', 'qt', 'r', 'react-native', 'reactjs', 'regex', 'rest', 'ruby', 'ruby-on-rails',
        'ruby-on-rails-3', 'scala', 'security', 'selenium', 'shell', 'spring', 'spring-boot', 'spring-mvc', 'sql',
        'sql-server', 'ssl', 'string', 'swift', 't-sql', 'templates', 'testing', 'twitter-bootstrap', 'typescript',
        'ubuntu', 'unit-testing', 'unix', 'user-interface', 'validation', 'visual-studio', 'visual-studio-2010',
        'web-services', 'webpack', 'windows', 'winforms', 'wpf', 'x86', 'xaml', 'xcode', 'xml']


##########################
#       Fonctions        #
##########################

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
    model_path = f"{repertoire_projet}/gestion_modeles/mlruns/134896038366573457/8ed5fd432c104d9ba196a3dde3499664/artifacts/Best_Model"
    predic_model = mlflow.sklearn.load_model(model_path)

    return predic_model



@st.cache_resource
def use_and_predict(data, _emb_model, _pred_model):
    """

    :param data: données à prédire sous forme de liste, Serie
    :return: prediction des Tags
    """

    data_vec = _emb_model(data).numpy()
    data_pred = pd.Series(_pred_model.predict(data_vec)[0], index=Tags)

    pred = data_pred[data_pred == 1].index

    return list(pred)


