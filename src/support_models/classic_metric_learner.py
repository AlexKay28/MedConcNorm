import os
from abc import ABCMeta, abstractmethod, abstractproperty

# CONFIGS
N_JOBS = 5
RANDOM_CEED = 32

# select_features_wrapper
MLTOOL_NAME = "NCA"
MLTOOL_ARGS = {'max_iter': 500}

class MetricLearnTool(metaclass=ABCMeta):

    @abstractmethod
    def fit():
        """ fitting """

    @abstractmethod
    def transform():
        """ transform X"""

    @abstractmethod
    def fit_transform():
        """ fit and transform X inplace"""


class StandardSupervisedMetricLearner(MetricLearnTool):
    from metric_learn import (
        NCA, LFDA, LMNN, MLKR, RCA,
        ITML_Supervised, LSML_Supervised,
        MMC_Supervised, SDML_Supervised,
        RCA_Supervised, SCML_Supervised
    )
    def __init__(self, name, params):
        self.model = eval(f"self.{name}")(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X, y):
        return self.model.fit_transform(X, y)

class StandardWeaklySupervisedMetricLearner(MetricLearnTool):
    from metric_learn import (
        # ITML, LSML, MMC, SCML
        SDML
    )
    def __init__(self, name, params):
        self.model = eval(f"self.{name}")(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X, y):
        return self.model.fit_transform(X, y)

class StandardUnsupervisedMetricLearner(MetricLearnTool):
    from metric_learn import (
        Covariance
    )
    def __init__(self, name, params):
        self.model = eval(f"self.{name}")(**params)

    def fit(self, X, y):
        self.model.fit(X)

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X, y):
        return self.model.fit_transform(X)


class ClassicMetricLearner:
    __mltools_name = {
        # supervised
        'NCA':  StandardSupervisedMetricLearner,
        'LFDA': StandardSupervisedMetricLearner,
        'LMNN': StandardSupervisedMetricLearner,
        'MLKR': StandardSupervisedMetricLearner,
        'RCA':  StandardSupervisedMetricLearner,
        'ITML_Supervised': StandardSupervisedMetricLearner,
        'LSML_Supervised': StandardSupervisedMetricLearner,
        'MMC_Supervised':  StandardSupervisedMetricLearner,
        #'SDML_Supervised': StandardSupervisedMetricLearner, (temporarily removed)
        #'RCA_Supervised':  StandardSupervisedMetricLearner, (temporarily removed)
        'SCML_Supervised': StandardSupervisedMetricLearner,

        # weakly supervised
        #"SDML": StandardWeaklySupervisedMetricLearner, (temporarily removed)

        # unsupervised
        "Covariance": StandardUnsupervisedMetricLearner,
    }

    def __init__(self, mltool_name=MLTOOL_NAME, mltool_params=MLTOOL_ARGS, n_jobs=N_JOBS):
        self.mltool_name = mltool_name
        self.mltool_params = mltool_params if mltool_params is not None else {}
        self.model = self.__mltools_name[self.mltool_name](self.mltool_name, self.mltool_params)
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.model.fit(X, y)

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X, y):
        return self.model.fit_transform(X, y)

    @classmethod
    def available_mltools(self):
        return self.__mltools_name.keys()
