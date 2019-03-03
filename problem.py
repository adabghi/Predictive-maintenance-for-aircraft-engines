from __future__ import division, print_function
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, recall_score, precision_score
import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from sklearn.model_selection import  StratifiedKFold


problem_title = 'Predictive maintenance for aircraft engines'

# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------
workflow = rw.workflows.FeatureExtractorClassifier()

# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------
_target_column_name = 'labels'
_prediction_label_names = [0, 1, 2, 3]

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------


class MultiClassLogLoss(BaseScoreType):
    # subclass BaseScoreType to use raw y_pred (proba's)
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='mc_ll', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        score = log_loss(y_true, y_pred)
        return score


class WeightedPrecision(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='w_prec', precision=2):
        self.name = name
        self.precision = precision
        self.weights = [0.1, 0.2, 0.3, 0.4]

    def __call__(self, y_true_label_index, y_pred_label_index):
        class_score = precision_score(y_true_label_index, y_pred_label_index, average=None)
        score = np.sum(class_score * self.weights)
        return score


class WeightedRecall(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='w_rec', precision=2):
        self.name = name
        self.precision = precision
        self.weights = [0.5, 0.3, 0.1, 0.1]

    def __call__(self, y_true_label_index, y_pred_label_index):
        class_score = recall_score(y_true_label_index, y_pred_label_index, average=None)
        score = np.sum(class_score * self.weights)
        return score


class MacroAveragedF1(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='mixed', precision=2):
        self.name = name
        self.precision = precision
        self.weighted_recall = WeightedRecall()
        self.weighted_precision = WeightedPrecision()

    def __call__(self, y_true, y_pred):
        rec = self.weighted_recall(y_true, y_pred)
        prec = self.weighted_precision(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec + 10 ** -15)


class Mixed(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='mixed', precision=2):
        self.name = name
        self.precision = precision
        self.macro_averaged_f1 = MacroAveragedF1()
        self.multi_class_log_loss = MultiClassLogLoss()

    def __call__(self, y_true, y_pred):
        y_chap = np.zeros((y_pred.shape))
        y_chap[np.arange(len(y_chap)), np.argmax(y_pred, axis=1)] = 1

        f1 = self.macro_averaged_f1(y_true, y_chap)
        ll = self.multi_class_log_loss(y_true, y_pred)
        return ll + (1 - f1)


score_types = [
    # mixed log-loss/f1 score
    Mixed(),
    # log-loss
    MultiClassLogLoss(),
    # weighted precision and recall ( over all classes)
    WeightedPrecision(),
    WeightedRecall(),
]


# -----------------------------------------------------------------------------
# Cross-validation scheme
# -----------------------------------------------------------------------------

def get_cv(X, y):
    cv = StratifiedKFold(n_splits=5,  random_state=57)
    return cv.split(X, y)

# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------


def _read_data(path, type_):
    fname1 = '{}_FD001.txt'.format(type_)
    fname2 = '{}_FD003.txt'.format(type_)

    fp1 = os.path.join(path, 'data', fname1)
    fp2 = os.path.join(path, 'data', fname2)

    df1 = pd.read_csv(fp1, sep=' ', header=None)
    df2 = pd.read_csv(fp2, sep=' ', header=None)
    if type_ == 'test':
        labels1 = 'RUL_FD001.txt'
        plabels1 = os.path.join(path, 'data', labels1)
        labels2 = 'RUL_FD003.txt'
        plabels2 = os.path.join(path, 'data', labels2)
        target_test1 = pd.read_csv(plabels1, sep=' ', header=None)
        target_test2 = pd.read_csv(plabels2, sep=' ', header=None)
        target_test2.index = range(len(target_test1), len(target_test1) + len(target_test2))
        target_test = pd.concat([target_test1, target_test2])
        target_test.drop([1], axis=1, inplace=True)
        target_test.reset_index(level=0, inplace=True)
        target_test.columns = ['ID', 'ttf']
        target_test['ID'] += 1

    #Merging data
    df2[0] = df2[0] + df1[0].max()
    frames = [df1, df2]
    data = pd.concat(frames)
    data.reset_index(level=0, inplace=True)
    data.drop(['index'], axis=1, inplace=True)

    data.drop([26, 27], axis=1, inplace=True)
    column_name = ['ID', 'Cycle', 'op_set_1', 'op_set_2', 'op_set_3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
                   's9',
                   's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    data.columns = column_name

    ## TTF calculation

    if type_ == 'train':
        failure_cycle = pd.DataFrame(data.groupby('ID')['Cycle'].max())
        failure_cycle.reset_index(level=0, inplace=True)
        failure_cycle.columns = ['ID', 'ttf']

        data_merged = pd.merge(data, failure_cycle, on='ID')
        data_merged['ttf'] = data_merged['ttf'] - data_merged['Cycle']

    if type_ == 'test':
        max_cycle = pd.DataFrame(data.groupby('ID')['Cycle'].max())
        max_cycle.reset_index(level=0, inplace=True)
        max_cycle.columns = ['ID', 'max_cycle']
        data_merged = pd.merge(data, target_test, on='ID')
        data_merged = pd.merge(data_merged, max_cycle, on='ID')
        data_merged['ttf'] = data_merged['max_cycle'] - data_merged['Cycle'] + data_merged['ttf']

    ## labeling data
    data_merged['labels'] = data_merged['ttf'].apply(lambda x: 0 if x <= 10 else 1 if x <= 30 else 2 if x <= 100 else 3)
    y = data_merged['labels']

    # for the "quick-test" mode, use less data
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        N_small = 5000
        data = data[:N_small]
        y = y[:N_small]

    return data, y


def get_train_data(path='.'):
    return _read_data(path, 'train')


def get_test_data(path='.'):
    return _read_data(path, 'test')
