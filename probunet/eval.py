import inspect
import numpy as np


def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)



class ConfusionMatrix:
    """Helper class to hold confusion matrix values, so we don't have to recompute when evaluating multiple metrics."""

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.product(self.reference.shape))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full



class Evaluator:
    """Object that holds test and reference segmentations with label information
    and computes a number of metrics on the two. 'labels' must either be an
    iterable of numeric values (or tuples thereof) or a dictionary with string
    names and numeric values.
    """

    default_metrics = [
        "False Positive Rate",
        "Dice",
        "Jaccard",
        "Precision",
        "Recall",
        "Accuracy",
        "False Omission Rate",
        "Negative Predictive Value",
        "False Negative Rate",
        "True Negative Rate",
        "False Discovery Rate",
        "Total Positives Test",
        "Total Positives Reference"
    ]

    # this is for metrics that take a long time to compute (e.g. Hausdorff from medpy)
    default_advanced_metrics = []

    def __init__(self,
                 test=None,
                 reference=None,
                 label_values=None,
                 label_names=None,
                 metrics=None,
                 advanced_metrics=None,
                 nan_for_nonexisting=True):

        self.test = None
        self.reference = None
        self.confusion_matrix = ConfusionMatrix()
        self.label_values = None
        self.label_names = None
        self.nan_for_nonexisting = nan_for_nonexisting
        self.result = None

        self.metrics = []
        if metrics is None:
            for m in self.default_metrics:
                self.metrics.append(m)
        else:
            for m in metrics:
                self.metrics.append(m)

        self.advanced_metrics = []
        if advanced_metrics is None:
            for m in self.default_advanced_metrics:
                self.advanced_metrics.append(m)
        else:
            for m in advanced_metrics:
                self.advanced_metrics.append(m)

        self.set_reference(reference)
        self.set_test(test)
        if label_values is not None:
            self.set_labels(label_values, label_names)
        else:
            if test is not None and reference is not None:
                self.construct_labels()

    def set_test(self, test):
        """Set the test segmentation."""

        self.test = test

    def set_reference(self, reference):
        """Set the reference segmentation."""

        self.reference = reference

    def set_labels(self, label_values, label_names=None):
        """Set the labels.
        :param labels= may be a dictionary (int->str), a set (of ints), a tuple (of ints) or a list (of ints). Labels
        will only have names if you pass a dictionary"""

        self.label_values = label_values
        if label_names is not None:
            self.set_label_names(label_names)

    def set_label_names(self, label_names):

        self.label_names = label_names

    def construct_labels(self):
        """Construct label set from unique entries in segmentations."""

        if self.test is None and self.reference is None:
            raise ValueError("No test or reference segmentations.")
        elif self.test is None:
            label_values = np.unique(self.reference)
        else:
            label_values = np.union1d(np.unique(self.test),
                                      np.unique(self.reference))
        self.label_values = list(map(lambda x: int(x), label_values))

    def set_metrics(self, metrics):
        """Set evaluation metrics"""

        if isinstance(metrics, set):
            self.metrics = list(metrics)
        elif isinstance(metrics, (list, tuple, np.ndarray)):
            self.metrics = metrics
        else:
            raise TypeError("Can only handle list, tuple, set & numpy array, but input is of type {}".format(type(metrics)))

    def add_metric(self, metric):

        if metric not in self.metrics:
            self.metrics.append(metric)

    def evaluate(self, test=None, reference=None, advanced=False, **metric_kwargs):
        """Compute metrics for segmentations."""
        if test is not None:
            self.set_test(test)

        if reference is not None:
            self.set_reference(reference)

        if self.test is None or self.reference is None:
            raise ValueError("Need both test and reference segmentations.")

        if self.label_values is None:
            self.construct_labels()

        self.metrics.sort()

        # get functions for evaluation
        # somewhat convoluted, but allows users to define additonal metrics
        # on the fly, e.g. inside an IPython console
        _funcs = {m: ALL_METRICS[m] for m in self.metrics + self.advanced_metrics}
        frames = inspect.getouterframes(inspect.currentframe())
        for metric in self.metrics:
            for f in frames:
                if metric in f[0].f_locals:
                    _funcs[metric] = f[0].f_locals[metric]
                    break
            else:
                if metric in _funcs:
                    continue
                else:
                    raise NotImplementedError(
                        "Metric {} not implemented.".format(metric))

        # get results
        self.result = {}

        eval_metrics = self.metrics
        if advanced:
            eval_metrics += self.advanced_metrics

        for i, label_value in enumerate(self.label_values):
            if self.label_names is not None:
                name = self.label_names[label_value]
            else:
                name = str(label_value)
            self.result[name] = {}
            if not hasattr(label_value, "__iter__"):
                self.confusion_matrix.set_test(self.test == label_value)
                self.confusion_matrix.set_reference(self.reference == label_value)
            else:
                current_test = 0
                current_reference = 0
                for l in label_value:
                    current_test += (self.test == l)
                    current_reference += (self.reference == l)
                self.confusion_matrix.set_test(current_test)
                self.confusion_matrix.set_reference(current_reference)
            for metric in eval_metrics:
                self.result[name][metric] = _funcs[metric](confusion_matrix=self.confusion_matrix,
                                                           nan_for_nonexisting=self.nan_for_nonexisting,
                                                           **metric_kwargs)

        return self.result

    def to_dict(self):

        if self.result is None:
            self.evaluate()
        return self.result

    def to_array(self):
        """Return result as numpy array (labels x metrics)."""

        if self.result is None:
            self.evaluate()

        result_metrics = sorted(self.result[list(self.result.keys())[0]].keys())

        a = np.zeros((len(self.label_values), len(result_metrics)), dtype=np.float32)

        for i, label in enumerate(self.label_values):
            if self.label_names is not None:
                label = self.label_names[label]
            for j, metric in enumerate(result_metrics):
                a[i][j] = self.result[label][metric]

        return a



def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))


def jaccard(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp + fn))


def precision(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp))


def sensitivity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fn))


def recall(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    return sensitivity(test, reference, confusion_matrix, nan_for_nonexisting, **kwargs)


def specificity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tn / (tn + fp))


def accuracy(test=None, reference=None, confusion_matrix=None, **kwargs):
    """(TP + TN) / (TP + FP + FN + TN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + tn) / (tp + fp + tn + fn))


def fscore(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, beta=1., **kwargs):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""

    precision_ = precision(test, reference, confusion_matrix, nan_for_nonexisting)
    recall_ = recall(test, reference, confusion_matrix, nan_for_nonexisting)

    return (1 + beta*beta) * precision_ * recall_ /\
        ((beta*beta * precision_) + recall_)


def false_positive_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (FP + TN)"""

    return 1 - specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_omission_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TN + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(fn / (fn + tn))


def false_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TP + FN)"""

    return 1 - sensitivity(test, reference, confusion_matrix, nan_for_nonexisting)


def true_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    return specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_discovery_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (TP + FP)"""

    return 1 - precision(test, reference, confusion_matrix, nan_for_nonexisting)


def negative_predictive_value(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FN)"""

    return 1 - false_omission_rate(test, reference, confusion_matrix, nan_for_nonexisting)


def total_positives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fp


def total_negatives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fn


def total_positives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fn


def total_negatives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fp


ALL_METRICS = {
    "False Positive Rate": false_positive_rate,
    "Dice": dice,
    "Jaccard": jaccard,
    "Precision": precision,
    "Recall": recall,
    "Accuracy": accuracy,
    "False Omission Rate": false_omission_rate,
    "Negative Predictive Value": negative_predictive_value,
    "False Negative Rate": false_negative_rate,
    "True Negative Rate": true_negative_rate,
    "False Discovery Rate": false_discovery_rate,
    "Total Positives Test": total_positives_test,
    "Total Negatives Test": total_negatives_test,
    "Total Positives Reference": total_positives_reference,
    "total Negatives Reference": total_negatives_reference
}