from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    """
    Cette fonction renvoie la matrice de confusion et la courbe ROC pour un modèle 
    de classification binaire afin d'en juger les performances.

    :param model: modèle (scikit_learn) à evaluer
    :param X_test: données de test
    :param y_test: labels des données de test

    """
    # Prédiction sur les données de test
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Affichage de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    precision = cm / cm.sum(axis=0)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(precision, cmap='Blues')
    ax.set_xticks(range(len(precision)))
    ax.set_yticks(range(len(precision)))
    ax.set_xticklabels(["avis négatif", "avis positif"], fontsize=12)
    ax.set_yticklabels(["avis négatif", "avis positif"], fontsize=12)
    for i in range(len(precision)):
        for j in range(len(precision)):
            abs_value = cm[i][j]
            pct_value = cm[i][j] / cm[i].sum() * 100
            text = '{:.1f}%\n{}'.format(pct_value, abs_value)
            ax.annotate(text, xy=(j, i), ha='center', va='center', color='black',
                        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.7), fontsize=12)
    plt.xlabel('Prédiction', fontsize=18)
    plt.ylabel('Réalité', fontsize=18)
    plt.title('Matrice de confusion', fontsize=30, color='#2471a3')
    ax.title.set_position([.5, 200])
    plt.colorbar(im)
    plt.show()

    # Affichage de la courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=4)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs', fontsize=18)
    plt.ylabel('Taux de vrais positifs', fontsize=18)
    plt.title('Courbe ROC (Receiver Operating Characteristic)', fontsize=22, color="#0d270f")
    plt.legend(loc="lower right", fontsize=12)
    plt.show()
