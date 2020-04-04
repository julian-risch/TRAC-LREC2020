import numpy as np
import sklearn


class ExplainableClassifier:
    """
    An interface for classifiers that support attribution-based explanations for text categorizations.

    To implement this interface implement all methods that raise an NotImplementedError in this interface.
    """

    def __init__(self, tokenize=None, label_mapping=None, ignored_regexes=None):
        """ Constructor for ExplainableClassifier.

        :param tokenize: The tokenize function the classifier uses for preprocessing.
        :param label_mapping: The class labels in the form of a two dimensional array.
                              Labels should be reached by: label_mapping[label_number, class_number].
        :param ignored_regexes: A list of regexes that the classifier should ignore for text categorizations.
        """
        self.tokenize = tokenize
        self.label_mapping = label_mapping
        if ignored_regexes is None:
            self.ignored_regexes = []
        else:
            self.ignored_regexes = ignored_regexes

    def train(self, X, y):
        """ Trains the classifier on the the given dataset X with the given labels y.

               :param X: An array of texts the classifier should be trained on.
               :param y: An array of numeric class labels for X.
                         For multi-label classifications this array is two dimensional.
        """
        raise NotImplementedError

    def evaluate(self, X, y, metric='accuracy'):
        """ Evaluates the classifier on the given evaluation set X, y and the given evaluation metric.

        :param X: An array of texts the classifier should be evaluated on.
        :param y: An array of numeric class labels for X.
                  For multi-label classifications this array is two dimensional.
        :param metric: The evaluation metric. Either: 'accuracy' or 'f1' Default: 'accuracy'.
        :return: An array of that contains the accuracies if metric is 'accuracy'
                 or a tripels (precision, recall, f1) if metric is 'f1' for each label.
        """
        y_pred = self.predict(X)
        if y.ndim == 1:
            y = y[:, np.newaxis]

        if metric == 'accuracy':
            accuracies = [0] * y.shape[1]
            for label in range(y.shape[1]):
                accuracies[label] = sklearn.metrics.accuracy_score(y[:, label], y_pred[:, label])
            return accuracies
        elif metric == 'f1':
            f1 = [0] * y.shape[1]
            for label in range(y.shape[1]):
                if y[:, label].max() == 1:
                    f1[label] = sklearn.metrics.precision_recall_fscore_support(y[:, label], y_pred[:, label],
                                                                                average='binary')[:3]
                else:
                    f1[label] = sklearn.metrics.precision_recall_fscore_support(y[:, label], y_pred[:, label],
                                                                                average='macro')[:3]
            return f1
        else:
            raise ValueError('No implementation for the given metric.')

    def predict(self, X):
        """ Predicts the classes for the input texts X.

        :param X: An array of texts to predict the classes to.
        :return: A two dimensional array of predicted classes.
                 First dimension is the label (for multi-class classification).
                 Second dimension is the corresponding text index of the input X.
        """
        raise NotImplementedError

    def explain(self, x, method=None, label=0, class_to_explain=None, options=None):
        """ Generates an attribution based explanation for text categorizations.

                :param x: The text to classify and generate an explanation for.
                :param method: The explainability method to use.
                :param label: The label to explain.
                :param class_to_explain: The numeric class label to explain.
                :param options: Options for the explainability method.
                :param domain_mapping: Whether the relevance scores should be mapped onto the input text (True)
                                       or the document representation (False).
                :return: A list of dictionaries that represent the tokens of the texts with the assigned relevance scores,
                         if domain_mapping is true. Otherwise exports the relevance scores in a vector
                         with the same dimension of the document representation.
        """
        raise NotImplementedError

    def export_model(self, model):
        """ Exports the model to the given directory.

            A model is exported to a directory that contains the model ('classifier')
            and the parameters ('parameters.pkl') as files.

        :param model: The directory to export the model to.
        """
        raise NotImplementedError

    @classmethod
    def import_model(cls, model, *kargs, **kwargs):
        """ Imports a model from the given directory.

        :param model: The directory to the exported model.
        :return: The imported model.
        """
        raise NotImplementedError

    def preprocess_text(self, text, significant_only=True):
        """ Preprocesses the given text.

        :param text: The text to preprocess.
        :param significant_only: Whether the preprocessed text should be returned as list of not ignored tokens
                                 in string format or as a list of all (ignored and not ignored) tokens
                                 in dictionary format. Ignored tokens have the attribute ignore set to True.
        :return: A list of not ignored tokens in string format, if significant_only is true.
                 Otherwise a list of all (ignored and not ignored) tokens in dictionary format.
        """
        raise NotImplementedError
