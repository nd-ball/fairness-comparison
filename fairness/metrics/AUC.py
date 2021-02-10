from fairness.metrics.Metric import Metric
from sklearn.metrics import roc_auc_score

class AUC(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'AUC'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        return roc_auc_score(actual, predicted)
