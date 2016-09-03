"""
library for Early pediatric AKI prediction
author: 310248864, Ben ByungGu Cho
last modified: 20160707
"""

import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.cross_validation import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import random
from sklearn import metrics

import shutil
import tempfile

import matplotlib.pyplot as plt
from scipy import linalg, ndimage

# from sklearn.feature_extraction.image import grid_to_graph
from sklearn import feature_selection
from sklearn.cluster import FeatureAgglomeration
# from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.externals.joblib import Memory
from sklearn.cross_validation import KFold




import pedAKI_utilities as paki


class AKI_predictor_log():
    def __init__(self, data, ready=False, cutoff=0, fill_mode='mean', ref_type='onset', cv=None,
                 sm='accuracy', test_size=0.3, Cs=10, timelag=None, timewindow=None, pre_model=None,
                 do_balance=True):
        """

        :param data: predictor/response variable matrix for trainig/testing
        :param cutoff: column(feature) with non-null values less than cutoff*totalrows are removed.
                       row(patient) with non-null values less than cutoff*totalfeatures are removed.
        :param fill_mode: missing value filling mode, 'mean' or 'age_mean' or 'age_rand'
        :param ref_type: reference time mode, 'onset' or 'admit'
        :param cv: number of cross validation when training the model
        :param sm: scoring method, 'accuracy' or 'recall'
        :param test_size: proportion of test data set respective to the total data set
        :param Cs: number of regularizing strengths
        """


        self.timelag = timelag
        self.timewindow = timewindow
        self.cutoff=cutoff
        self.fill_mode = fill_mode
        self.ref_type = ref_type
        self.no_cv = cv
        self.scoring_mode=sm
        self.test_size = test_size
        self.Cs=10
        if ready == True:
            try:
                self.X_train = data['X_train']
                self.y_train = data['y_train']
            except:
                pass
            try:
                self.X_test = data['X_test']
                self.y_test = data['y_test']
            except:
                pass
            try:
                self.cols = data['predictors']
            except:
                pass
        else:
            io_mat = data.copy(deep=True)
            self.prepdata(io_mat, fill_mode, ref_type, do_balance=do_balance)
        # self.predictor = LogisticRegressionCV(Cs=Cs, cv=cv, scoring=sm, refit=True, class_weight='balanced')
        if pre_model is not None:
            self.predictor = pre_model
            self.pretrained = True
        else:
            self.predictor = LogisticRegressionCV(Cs=Cs, cv=cv, scoring=sm, refit=True)
            self.pretrained = False
        # self.predictor = LogisticRegression(Cs=Cs, cv=cv, scoring=sm, refit=True)

    # Balance AKI and control groups
    def balClass(self, X, y):

        y = pd.Series(y.AKI)
        X.reset_index(inplace=True)
        y.reset_index(inplace=True, drop=True)
        X_con = X.loc[y.values==0, :]
        y_con = y[y.values==0]
        no_control = len(X_con.index)
        pool = X.index[y.values > 0]
        # stop
        rand_idx = [random.choice(pool) for count in range(no_control)]
        X_aki = X.loc[rand_idx, :]
        y_aki = y[rand_idx]
        # X = pd.concat([X_con, X_aki], axis=0).as_matrix()
        # y = pd.concat([y_con, y_aki]).as_matrix()
        X = pd.concat([X_con, X_aki], axis=0)
        y = pd.concat([y_con, y_aki])
        X.drop(['index'], inplace=True, axis=1)
        return X, y


    def prepdata(self, io_mat, fill_mode, ref_type, do_balance=True):
        """
        Function that converts predictor/response variable matrix to an eligible form
        to use scikit learn for training/testing. Called by __init__()
        :param io_mat: predictor/response variable matrix
        :param fill_mode: missing value filling mode
        :param ref_type: reference time mode
        :return:
        """
        try:
            io_mat.drop(['race_unique'], inplace=True, axis=1)
        except:
            pass
        io_mat_filled = paki.fillMissing(io_mat, mode=fill_mode, ref_time=ref_type)
        sparse_col = np.array(io_mat.count()[io_mat.count().values < io_mat.count()['encounter_id'] * self.cutoff].index)
        io_mat_rmc = io_mat_filled.drop(sparse_col, axis=1)
        sparse_row = np.array(
            io_mat.count(axis=1)[io_mat.count(axis=1).values < (len(io_mat.columns) - 2) * self.cutoff].index)
        io_mat_final = io_mat_rmc.drop(sparse_row, axis=0)
        io_mat_final['AKI'] = (io_mat_final.AKI_stage > 0).astype(int)
        io_mat_final['sex_M'] = (io_mat_final.sex == 'M').astype(int)
        # io_mat_col =

        # Create the formula as an input to patsy dmatrices
        ex_ft = ['encounter_id', 'patient_id', 'AKI', 'AKI_stage', 'sex']
        in_ft = [ft for ft in io_mat_final.columns if ft not in ex_ft]
        y = pd.DataFrame(io_mat_final.AKI)
        X = io_mat_final.loc[:, in_ft]
        # no_feat = len(io_mat_final.columns) - 3
        # formula = "AKI ~"
        # for col, count in zip(in_ft, range(len(in_ft))):
        #     if col == 'sex':
        #         formula = formula + " + C(" + col + ")"
        #     elif count == 0:
        #         formula = formula + " " + col
        #     else:
        #         formula = formula + " + " + col
        #
        # y, X = dmatrices(formula, io_mat_final, return_type="dataframe")
        # X.rename(columns={'C(sex)[T.M]': 'sex_M'}, inplace=True)
        self.cols = list(X.columns.values)

        X_train, X_test, y_train, y_test \
            = train_test_split(X, y, test_size=self.test_size, random_state=0)
        if do_balance:
            X_train, y_train = self.balClass(X_train, y_train)
            X_test, y_test = self.balClass(X_test, y_test)

        self.X_train = X_train
        self.y_train = np.ravel(y_train.as_matrix())
        self.X_test = X_test
        self.y_test = np.ravel(y_test.as_matrix())




    def cv_full(self, do_test=True):
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        X_train = scaler.transform(self.X_train)
        try:
            X_test = scaler.transform(self.X_test)
            print('Test set normalized ...')
        except:
            pass

        if self.pretrained==True:
            print('Already trained ...')
        else:
            print('training ...')
            self.predictor.fit(X_train, self.y_train)
        if do_test:
            print('testing ...')
            self.y_pred = self.predictor.predict(X_test)
            self.prob_pred = self.predictor.predict_proba(X_test)
            # if self.scoring_mode == 'accuracy':
            #     self.err_train = 1 - np.mean(self.predictor.scores_[1.0])
            #     self.sens_train = None
            #     self.spec_train = None
            #
            # elif self.scoring_mode == 'recall':
            #     y_pred_train = self.predictor.predict(X_train)
            #     tp_train = np.sum((y_pred_train > 0) & (self.y_train > 0))
            #     tn_train = np.sum((y_pred_train == 0) & (self.y_train == 0))
            #     fp_train = np.sum((y_pred_train > 0) & (self.y_train == 0))
            #     fn_train = np.sum((y_pred_train == 0) & (self.y_train > 0))
            #     sn_train = float(tp_train) / float(tp_train + fn_train)
            #     sp_train = float(tn_train) / float(tn_train + fp_train)
            #     self.sens_train = sn_train
            #     self.spec_train = sp_train
            #     self.err_train = None

            self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
            self.auc = metrics.roc_auc_score(self.y_test, self.prob_pred[:, 1])
            print(self.auc)

            self.tp = np.sum((self.y_pred > 0) & (self.y_test > 0))
            self.tn = np.sum((self.y_pred == 0) & (self.y_test == 0))
            self.fp = np.sum((self.y_pred > 0) & (self.y_test == 0))
            self.fn = np.sum((self.y_pred == 0) & (self.y_test > 0))
            try:
                self.sensitivity = float(self.tp) / float(self.tp + self.fn)
            except:
                self.sensitivity = None
                # Specificity
            try:
                self.specificity = float(self.tn) / float(self.tn + self.fp)
            except:
                self.specificity = None
            # Positive predictive value
            try:
                self.ppv = float(self.tp) / float(self.tp + self.fp)
            except:
                self.ppv = None
            # Negative predictive value
            try:
                self.npv = float(self.tn) / float(self.tn + self.fn)
            except:
                self.npv = None
            # Positive likelihood ratio
            try:
                self.plr = self.sensitivity / (1 - self.specificity)
            except:
                self.plr = None
            # Negative likelihood ratio
            try:
                self.nlr = (1 - self.sensitivity) / self.specificity
            except:
                self.nlr = None
            # Get coefficients in the order of significance
            self.coeff = self.sort_coeff(self.cols, np.hstack(self.predictor.coef_))
            self.fpr, self.tpr, self.thresholds \
                = metrics.roc_curve(self.y_test, self.prob_pred[:, 1])

    def get_full(self):

        # stop

        return {'predictor': self.predictor,
                'predictor variables': self.cols,
                # 'training error': self.err_train,
                # 'training sensitivity': self.sens_train,
                # 'training specificity': self.spec_train,
                'test accuracy': self.accuracy,
                'test auc': self.auc,
                'test specificity': self.specificity,
                'test sensitivity': self.sensitivity,
                'test ppv': self.ppv,
                'test npv': self.npv,
                'test plr': self.plr,
                'test nlr': self.nlr,
                'roc': (self.fpr, self.tpr, self.thresholds)}

    def sort_coeff(self, cols, coeff):
        clf_coeff = pd.DataFrame(zip(cols, coeff))
        clf_coeff.rename(columns={0: 'label', 1: 'coeff'}, inplace=True)
        clf_coeff['abs'] = clf_coeff.coeff.abs()
        clf_coeff.sort_values(by='abs', ascending=False, inplace=True)
        clf_coeff.drop('abs', axis=1, inplace=True)
        return clf_coeff


    def get_coeff_full(self):
        return self.coeff
    def get_coeff_compact(self):
        return self.coeff_small

    def choose_pv(self, auc_th=0.0001, n_clusters=22):
        grouped_predictors, grouped_idx = self.agglomeration(n_clusters=n_clusters)

        acc_group_idx = list([])
        acc_group_name = list([])
        for ingroup_idx, ingroup_label in zip(grouped_idx, grouped_predictors):
            best_idx, best_name = self.auc_within(ingroup_idx, ingroup_label)

            if best_idx is not None:
                acc_group_idx.append(best_idx)
                acc_group_name.append(best_name)
        self.auc_drop(auc_th=auc_th, group=acc_group_idx, group_name=acc_group_name)

    def agglomeration(self, n_clusters):
        cv = KFold(len(self.y_train), 5)
        agglo = FeatureAgglomeration(n_clusters=n_clusters)
        agglo.fit(self.X_train)
        grouped_names = list([])
        grouped_idx = list([])
        col_idx = np.arange(len(self.cols))
        for label in np.unique(agglo.labels_):
            group = [name for (name, group_id) in zip(self.cols, agglo.labels_) if group_id==label]
            group_idx = [idx for (idx, group_id) in zip(col_idx, agglo.labels_) if group_id==label]
            grouped_names.append(group)
            grouped_idx.append(group_idx)
        return grouped_names, grouped_idx

    def auc_within(self, ingroup_idx, ingroup_name):
        predictor = LogisticRegressionCV(Cs=self.Cs, cv=self.no_cv,
                                                  scoring=self.scoring_mode, refit=True)
        col_idx = np.arange(len(self.cols))
        mask = ~np.in1d(col_idx, ingroup_idx)
        X_train = self.X_train[:, mask]
        X_test = self.X_test[:, mask]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        predictor.fit(X_train, self.y_train)
        prob_pred = predictor.predict_proba(X_test)
        auc_ref = metrics.roc_auc_score(self.y_test, prob_pred[:,1])
        best_auc = auc_ref
        best_idx = None
        best_name = None

        for idx, name in zip(ingroup_idx, ingroup_name):
            mask_idx = (col_idx==idx)
            mask_in = (mask | mask_idx)
            X_train_in = self.X_train[:, mask_in]
            X_test_in = self.X_test[:, mask_in]

            scaler = StandardScaler()
            scaler.fit(X_train_in)
            X_train_in = scaler.transform(X_train_in)
            X_test_in = scaler.transform(X_test_in)

            predictor.fit(X_train_in, self.y_train)
            prob_pred_in = predictor.predict_proba(X_test_in)
            auc_in = metrics.roc_auc_score(self.y_test, prob_pred_in[:,1])
            if auc_in > best_auc:
                best_auc = auc_in
                best_idx = idx
                best_name = name

        print(best_name)

        return best_idx, best_name



    def auc_drop(self, auc_th, group=None, group_name=None):
        self.auc_decrease_th = auc_th
        if group is None:
            X_train_all = pd.DataFrame(self.X_train, columns=self.cols)
            X_test_all = pd.DataFrame(self.X_test, columns=self.cols)
        else:
            X_train_all = pd.DataFrame(self.X_train[:,group], columns=group_name)
            X_test_all = pd.DataFrame(self.X_test[:,group], columns=group_name)

        predictors = list()

        for col in X_train_all.columns:
            X_train_drop = X_train_all.drop(col, axis=1)
            X_test_drop = X_test_all.drop(col, axis=1)

            scaler = StandardScaler()
            scaler.fit(X_train_drop)
            X_train_drop = scaler.transform(X_train_drop)
            X_test_drop = scaler.transform(X_test_drop)

            predictor_drop = LogisticRegressionCV(Cs=self.Cs, cv=self.no_cv,
                                                  scoring=self.scoring_mode, refit=True)
            predictor_drop.fit(X_train_drop, self.y_train)
            # y_pred_drop = predictor_drop.predict(X_test_drop)
            prob_pred_drop = predictor_drop.predict_proba(X_test_drop)
            auc = metrics.roc_auc_score(self.y_test, prob_pred_drop[:, 1])
            # print('auc drop by excluding {}: {}'.format(col, self.auc-auc))
            if self.auc - auc > auc_th:
                # print('{} is important'.format(col))
                predictors.append(col)
            else:
                pass

        self.X_train_small = X_train_all.loc[:, predictors].as_matrix()
        self.X_test_small = X_test_all.loc[:, predictors].as_matrix()
        self.cols_small = predictors
        if self.ref_type == 'onset':
            print("reference time:{} ; time lag:{} ; time window:{}".
                  format(self.ref_type, int(-self.timelag), int(self.timewindow)))
        elif self.ref_type == 'admit':
            print("reference time:{} ; time lag:{} ; time window:{}".
                  format(self.ref_type, int(self.timelag), int(self.timewindow)))
        print('predictors(auc drop threshold={}):{}'.format(auc_th, predictors))


    def cv_compact(self):
        self.predictor_small = LogisticRegressionCV(Cs=self.Cs, cv=self.no_cv,
                                                    scoring=self.scoring_mode, refit=True)


        self.predictor_small.fit(self.X_train_small, self.y_train)
        self.y_pred_small = self.predictor_small.predict(self.X_test_small)
        self.prob_pred_small = self.predictor_small.predict_proba(self.X_test_small)

        if self.scoring_mode == 'accuracy':
            self.err_train_small = 1 - np.mean(self.predictor_small.scores_[1.0])
            self.sens_train_small = None
            self.spec_train_small = None

        elif self.scoring_mode == 'recall':
            y_pred_train_small = self.predictor_small.predict(self.X_train_small)
            tp_train_small = np.sum((y_pred_train_small > 0) & (self.y_train > 0))
            tn_train_small = np.sum((y_pred_train_small == 0) & (self.y_train == 0))
            fp_train_small = np.sum((y_pred_train_small > 0) & (self.y_train == 0))
            fn_train_small = np.sum((y_pred_train_small == 0) & (self.y_train > 0))
            sn_train_small = float(tp_train_small) / float(tp_train_small + fn_train_small)
            sp_train_small = float(tn_train_small) / float(tn_train_small + fp_train_small)
            self.sens_train_small = sn_train_small
            self.spec_train_small = sp_train_small
            self.err_train_small = None

        self.accuracy_small = metrics.accuracy_score(self.y_test, self.y_pred_small)

        self.auc_small = metrics.roc_auc_score(self.y_test, self.prob_pred_small[:, 1])

        self.tp_small = np.sum((self.y_pred_small > 0) & (self.y_test > 0))
        self.tn_small = np.sum((self.y_pred_small == 0) & (self.y_test == 0))
        self.fp_small = np.sum((self.y_pred_small > 0) & (self.y_test == 0))
        self.fn_small = np.sum((self.y_pred_small == 0) & (self.y_test > 0))
        try:
            self.sensitivity_small = float(self.tp_small) / float(self.tp_small + self.fn_small)
        except:
            self.sensitivity_small = None
        # Specificity
        try:
            self.specificity_small = float(self.tn_small) / float(self.tn_small + self.fp_small)
        except:
            self.specificity_small = None
        # Positive predictive value
        try:
            self.ppv_small = float(self.tp_small) / float(self.tp_small + self.fp_small)
        except:
            self.ppv_small = None
        # Negative predictive value
        try:
            self.npv_small = float(self.tn_small) / float(self.tn_small + self.fn_small)
        except:
            self.npv_small = None
        # Positive likelihood ratio
        try:
            self.plr_small = self.sensitivity_small / (1 - self.specificity_small)
        except:
            self.plr_small = None
        # Negative likelihood ratio
        try:
            self.nlr_small = (1 - self.sensitivity_small) / self.specificity_small
        except:
            self.nlr_small = None
        self.coeff_small = self.sort_coeff(self.cols_small, np.hstack(self.predictor_small.coef_))
        self.fpr_small, self.tpr_small, self.thresholds_small \
            = metrics.roc_curve(self.y_test, self.prob_pred_small[:, 1])

    def get_compact(self):
        return {'predictor': self.predictor_small,
                'predictor variables': self.cols_small,
                'training error': self.err_train_small,
                'training sensitivity': self.sens_train_small,
                'training specificity': self.spec_train_small,
                'test accuracy': self.accuracy_small,
                'test auc': self.auc_small,
                'test specificity': self.specificity_small,
                'test sensitivity': self.sensitivity_small,
                'test ppv': self.ppv_small,
                'test npv': self.npv_small,
                'test plr': self.plr_small,
                'test nlr': self.nlr_small,
                'roc': (self.fpr_small, self.tpr_small, self.thresholds_small)}

















