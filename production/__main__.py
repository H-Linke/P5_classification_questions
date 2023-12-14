import pandas as pd

from utils.fonctions import *
from utils.constantes import Tags


##########################
#      Variables         #
##########################


# Chargement des modèles d'embedding et de prédiction
use = use_embedding_model()
pred_model = chargement_models_predict()
scaler = chargement_scaler()
ACP = chargement_pca()

st.divider()
st.title('Proposition automatique de tags pour les questions postées sur StackOverFlow')
st.divider()

titre = st.text_input('Titre de la question :')
corpus = st.text_input('Question correspondante :')

ready = st.button('Lancer la proposition de tags :point_right:')
st.text("")
st.text("")
st.text("")

if ready:

    question = pd.Series([f'{titre} {corpus}'])
    question_norm = step_1_3_main_normalisation_texte(question)

    pred = use_and_predict(question_norm, use, scaler, ACP, pred_model)

    st.subheader('Voici comment a été converti la question pour le modèle : ')
    st.text_area('Voici comment a été converti la question pour le modèle : ', question_norm[0], label_visibility='collapsed')

    st.text("")
    st.subheader('Et voici les propositions de tags correspondants a la question : ')
    st.write(pred)


