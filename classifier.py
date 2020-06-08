import pandas as pd
import numpy as np
import scipy.stats as ss
from logistic_regression import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, auc

def modelInfo(model, train_X, train_Y, features_list, intercept, verbose=False):
    beta0 = 0 if not intercept else model.beta[0]
    beta = model.beta[1:] if intercept else model.beta
    e = (train_Y).astype(np.float32) - beta0
    for i in range(train_X[features_list].shape[1]):
        e -= beta[i] * train_X[features_list[i]].values
    n = train_X[features_list].shape[0]
    k = train_X[features_list].shape[1]
    RSS = np.sum(e ** 2)
    RSE = np.sqrt(RSS / (n - k))
    X = np.hstack((np.ones((n, 1)), train_X[features_list]))
    B = RSE ** 2 * np.linalg.inv(X.T @ X)
    se = []
    for i in range(k + 1):
        se.append(np.sqrt(B[i, i]))
    def pvalue(t_score):
        cdf = ss.norm.cdf(t_score)
        return 2 * min(cdf, 1 - cdf)
    beta = np.insert(beta, 0, beta0)
    head = ('Name', 'Coefficient', 'Std. error', 'norm_statistic', 'p_value')
    if verbose:
        print('{:<42}|{:^15s}|{:^15s}|{:^15s}|{:^15s}|'.format(*head))
    pval = 0.0
    max_col = ''
    intercept_ = intercept
    for i in range(k + 1):
        if i != 0:
            col = train_X[features_list].columns[i - 1]
        else:
            col = 'Intercept'
        if pval < pvalue(beta[i] / se[i]) and i > 0:
            pval = max(pvalue(beta[i] / se[i]), pval)
            max_col = col
        if pvalue(beta[i] / se[i]) > 0.05 and i == 0 and intercept:
            intercept_ = False
        b0_row =(col, beta[i], se[i], beta[i] / se[i], pvalue(beta[i] / se[i]))
        if verbose:
            print('{:<42}|{:^15f}|{:^15f}|{:^15f}|{:^15f}|'.format(*b0_row))
    return pval, max_col, intercept_


def repeat(train_X, train_Y, num, feature_list, intercept):
    for i in train_X.index.values:
        for col in train_X.columns:
            if pd.isna(train_X.loc[i][col]):
                train_X.loc[i, col] = train_X[col].mean()

    models = []
    for i in range(3):
        lr = LogisticRegression(fit_intercept=intercept)
        lr.fit(train_X[feature_list], train_Y.values[:, i])
        models.append(lr)
    model = models[num]
    return modelInfo(model, train_X, train_Y.values[:, num], feature_list, intercept)

def exclude_parameters(X, Y, num, all_features):
    k = len(all_features)
    intercept = True
    all_features_ = all_features.copy()
    for j in range(k):
        X = X[all_features_]
        pval, max_col, intercept = repeat(X, Y, num, all_features_, intercept)
        if pval <= 0.05:
            break
        if pval > 0.05:
            all_features_.remove(max_col)
    return all_features_, intercept

def find_params(X, Y, num_classes, all_features):
    all_features_ = []
    interceptions = []
    for i in range(num_classes):
        ex, intercept = exclude_parameters(X, Y, i, all_features)
        all_features_.append(ex.copy())
        interceptions.append(intercept)
    return all_features_.copy(), interceptions


class OnevsRestClassifier():
    def __init__(self, model, params = dict()):
        self.num_classes = 0
        self.models = []
        self.model = model
        self.params = params
        self.feature_lists = []
        self.interceptions = []
        self.X = None
        self.Y = None
        self.fraction = 0.8

    def impute(self, X=None, Y=None, frac=-1):
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        if frac == -1:
            frac = int(self.fraction * X.shape[0])
        train_X, test_X = X[:frac].copy(), X[frac:].copy()
        train_Y, test_Y = Y[:frac].copy(), Y[frac:].copy()
        for i in train_X.index.values:
            for col in train_X.columns:
                if pd.isna(train_X[col].loc[i]):
                    train_X.loc[i, col] = train_X[col].mean()

        for i in test_X.index.values:
            for col in test_X.columns:
                if pd.isna(test_X.loc[i][col]):
                    test_X.loc[i, col] = test_X[col].mean()
        return train_X, train_Y, test_X, test_Y

    def fit(self, X, Y, fraction = 0.8):
        frac = int(fraction * X.shape[0])
        train_Y, test_Y = Y[:frac].copy(), Y[frac:].copy()
        if sum(test_Y.values[:, 0]) == 0 or sum(test_Y.values[:, 1]) == 0 or sum(test_Y.values[:, 2]) == 0:
            print("All test labels are equal")
            return self.feature_lists, []
        train_X, train_Y, test_X, test_Y = self.impute(X, Y, frac)
        self.X, self.Y = X, Y
        self.num_classes = self.Y.shape[1]
        self.fraction = fraction
        all_features = list(self.X.columns.values)
        target = list(self.Y.columns.values)
        self.feature_lists, self.interceptions = find_params(train_X, train_Y[target], self.num_classes, all_features)
        
        print("Factors:", self.feature_lists)
        print("Use intercept: ", self.interceptions)
        for i in range(self.num_classes):
            lr = self.model(fit_intercept=self.interceptions[i], **self.params)
            lr.fit(train_X[self.feature_lists[i]], train_Y.values[:, i])
            self.models.append(lr)
            pred = lr.predict(test_X[self.feature_lists[i]].values)
            llf = log_loss(test_Y.values[:, i], pred)
            lr = self.model(fit_intercept=True, **self.params)
            lr.fit(None, train_Y.values[:, i])
            pred = lr.predict(test_X.shape[0])
            ll0 = log_loss(test_Y.values[:, i], pred)
            print("Model #{}".format(i + 1))
            print("McFadden's pseudo r^2: ", 1 - llf/ll0)
            stat = 2*(-llf + ll0)
            print("Statistics value:", stat)
            crit_value = ss.chi2.ppf(0.05, len(self.feature_lists[i]) + 1)
            print("Chi2 distribution critical value:", crit_value)
            if 1 - llf/ll0 < 0.01 or stat <= crit_value:
                print("Coefficient for model #" + str(i) + "is not significant.")
                return self.feature_lists, self.interceptions

        print("All coefficients were significant!")
        print('Fitted!')
        return self.feature_lists, self.interceptions

    def predict(self, X):
        preds = []
        for i in range(self.num_classes):
            lr = self.models[i]
            pred = lr.predict(X[self.feature_lists[i]].values)
            preds.append(pred.copy())
        preds = np.array(preds).T
        return preds

    def predict_proba(self, X):
        preds = []
        for i in range(self.num_classes):
            lr = self.models[i]
            pred = lr.predict_proba(X[self.feature_lists[i]].values)
            preds.append(pred.copy())
        preds = np.array(preds).T
        return preds
    
    def ROC_AUC_score(self, preds, test_Y):
        for i in range(self.num_classes):
            print("Roc-auc score for model #{}:".format(i), roc_auc_score(test_Y.values[:, i], preds[:, i], average='micro'))
        print("Roc-auc score for classifier:", roc_auc_score(test_Y.values, preds, average='micro'))

    def classifierModelsInfo(self):
        for i in range(self.num_classes):
            print("Model #{}".format(i + 1))
            frac = int(self.fraction * self.X.shape[0])
            train_X, train_Y, _, _ = self.impute(self.X, self.Y, frac)
            train_Y_ = train_Y.values[:, i]
            train_X = train_X.astype(np.float32)
            print(train_X)
            _, _, _ = modelInfo(self.models[i], train_X, train_Y_, self.feature_lists[i], self.interceptions[i], True) 