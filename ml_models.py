rom pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
from sklearn.metrics import roc_auc_score, f1_score

# Fetching UCI repo to load dataset
from ucimlrepo import fetch_ucirepo

def read_csv(id):
    phiusiil_phishing_url_website = fetch_ucirepo(id=id)

    # data (as pandas dataframes)
    X = phiusiil_phishing_url_website.data.features
    y = phiusiil_phishing_url_website.data.targets
    return X, y

X, y = read_csv(967)

# dropping categorical features
X = X.drop(['URL', 'Domain', 'TLD', 'Title'],axis=1)

# Test data split
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.33, random_state=1, shuffle=True)

models = []

models.append(('NB', GaussianNB()))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))

for name, model in models:
    print("\n\nModel name:", name)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print("\nAccuracy Score:",accuracy_score(Y_validation, predictions))
    print("Confusion matrix \n",confusion_matrix(Y_validation, predictions))
    auc = roc_auc_score(Y_validation, predictions)
    print("\nAUC-ROC Score:", auc)
    f1 = f1_score(Y_validation, predictions)


