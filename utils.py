import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import shap

class ModelPipeline:
    def __init__(self, n_folds:int=None, params=None, n_jobs:int=-1):
        self.n_folds = n_folds
        self.params = params
        self.n_jobs = n_jobs
        self.model = None
        self.best_model = None
    
    def set_LogisticRegression(self):
        self.model = LogisticRegression()
    
    def set_DecisionTree(self):
        self.model = DecisionTreeClassifier()
    
    def set_SVC(self):
        self.model = SVC()
    
    def set_RandomForest(self):
        self.model = RandomForestClassifier()
    
    def set_GradientBoosting(self):
        self.model = GradientBoostingClassifier()
    
    def fit_Model(self, X_train, y_train):
        if self.model:
            self.model.fit(X_train, y_train)
        else:
            raise ValueError("Model not initialised. Set a model first.")
    
    def predict_Model(self, X_test):
        if self.model:
            return self.model.predict(X_test)
        else:
            raise ValueError("Model not initialised. Set a model first.")

    def tune_GridSearch(self, X_train, y_train):
        if not self.model:
            raise ValueError("Model not initialised. Set a model first.")
        
        if self.params:
            grid_search = GridSearchCV(
                estimator=self.model, 
                param_grid=self.params, 
                cv=self.n_folds,
                n_jobs=self.n_jobs
                )
            grid_search.fit(X_train, y_train)
            self.best_model = grid_search.best_estimator_
            return grid_search.best_params_
        else:
            return None
    
    def tune_RandomSearch(self, X_train, y_train, n_iter=50):
        if not self.model:
            raise ValueError("Model not initialised. Set a model first.")

        if self.params:
            random_search = RandomizedSearchCV(
                estimator=self.model, 
                param_distributions=self.params, 
                n_iter=n_iter, 
                cv=self.n_folds,
                n_jobs=self.n_jobs
                )
            random_search.fit(X_train, y_train)
            self.best_model = random_search.best_estimator_
            return random_search.best_params_
        else:
            return None
    