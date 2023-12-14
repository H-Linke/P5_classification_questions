# Ce script contient toutes les fonctions liées à la récupération et au prétraitement des données

##########################
#        IMPORTS         #
##########################
import pandas as pd
import os
import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

from gestion_data.python_dev.constantes import emoticons_regex, stop_words, stop_words_2


##########################
#       Fonctions        #
##########################

def extraire_texte_html(texte_html):
    soup = BeautifulSoup(texte_html, 'html.parser')
    texte = soup.get_text()
    return texte


def replace_characters(text):
    text = re.sub(emoticons_regex, " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text


def step_0_formalisation(data_brut_path):
    """
    Cette fonction prend lis les 3 premiers fichiers csv et les renvoie sous forme de DataFrame pandas après avoir formalisé
    les colonnes notamment en retirant le formalisme HTML

    :param data_brut_path: commentaire sous forme de chaîne de caractères
    :return: dataframe formalisé
    """

    data_list = os.listdir(data_brut_path)

    data_01_path = os.path.join(data_brut_path, data_list[0])
    data_02_path = os.path.join(data_brut_path, data_list[1])
    data_03_path = os.path.join(data_brut_path, data_list[2])

    data_01 = pd.read_csv(data_01_path)
    data_02 = pd.read_csv(data_02_path)
    data_03 = pd.read_csv(data_03_path)

    data = pd.concat([data_01, data_02, data_03], ignore_index=True).drop_duplicates(ignore_index=True)

    data.Title = data.Title.apply(replace_characters)
    data.Body = data.Body.apply(extraire_texte_html).apply(replace_characters)

    # On concatène les colonnes Title et Body formant la question.
    data['Question'] = data.Title.str.cat(data.Body, sep=' ')

    return data.loc[:, ['Question', 'Tags']]


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


def step_1_3_main_normalisation_texte(data):
    """
    Cette fonction prend en entrée une Serie pandas dont elle normalise le texte
    Un DataFrame nettoyé et normalisé.

    :param data: Serie pandas contenant du texte
    :return: Serie pandas avec le texte normalisé
    """

    data = data.apply(step_1_1_clean_text) \
        .apply(step_1_2_lemmatize_text) \
        .apply(word_tokenize) \
        .apply(lambda comm: [word for word in comm if word not in stop_words_2]) \
        .apply(lambda x: ' '.join(x))

    return data


def step_2_preparation_tags(data, nbr_occ=40):
    """
    Cette fonction prend en entrée une Serie pandas dont elle ordonne les tags
    Un DataFrame nettoyé et normalisé.

    :param data: Serie pandas contenant les tags
    :param nbr_occ: nombre d'occurence minimum pour qu'un tag soit conservé
    :return: Serie pandas avec le texte normalisé
    """

    tags_hist = data.str.extractall(r'<(.*?)>').groupby(0).value_counts().sort_values()

    tags_autorises = tags_hist[tags_hist > nbr_occ].index

    data_filter = data.str.extractall(r'<(.*?)>').groupby(level=0)[0]\
        .apply(list).apply(lambda tags: [tag for tag in tags if tag in tags_autorises]).apply(' '.join)

    return data_filter, pd.Series(tags_autorises)

