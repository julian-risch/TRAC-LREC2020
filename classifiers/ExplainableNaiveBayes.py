import numpy as np
import re
import nltk
import os
import dill
from sklearn.feature_extraction.text import CountVectorizer

from ExplainableClassifier import ExplainableClassifier


class ExplainableNaiveBayes(ExplainableClassifier):
    """ A multinomial Naive Bayes text classifier as described in
        Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze,
        Introduction to Information Retrieval,
        Cambridge University Press. 2008.
        that can generate
        attribution-based explanations for text categoizations.
    """

    def __init__(self, tokenize=nltk.tokenize.word_tokenize, label_mapping=None,
                 ignored_regexes=None, stem=nltk.stem.snowball.EnglishStemmer().stem,
                 word_regex=r'[A-Za-z\']+'):
        """ Constructor for ExplainableSVM.

        :param tokenize: The tokenize function the Naive Bayes classifier uses for preprocessing.
        :param label_mapping: The class labels in the form of a two dimensional array.
                              Labels should be reached by: label_mapping[label_number, class_number].
        :param ignored_regexes: A list of regexes that the Naive Bayes classifier should ignore for text categorizations.
        :param word_regex: A regex that a token must match to be considered by the Naive Bayes classifier.
        :param stem: The stemming function the SVM uses for preprocessing of the text.
        """
        self.stem = stem
        self.word_regex = word_regex
        self.count_vectorizer = CountVectorizer(tokenizer=self.preprocess_text, analyzer='word',
                                                token_pattern='(?!spacechar|newlinechar)' + self.word_regex)
        self.p_c = None
        self.p_w_given_c = None
        self.p_c_given_w = None
        super().__init__(tokenize=tokenize, label_mapping=label_mapping,
                         ignored_regexes=ignored_regexes)

    def train(self, X, y):
        """ Trains the Naive Bayes classifier on the the given dataset X with the given labels y.

        :param X: An array of texts the Naive Bayes classifier should be trained on.
        :param y: An array of numeric class labels for X. For multi-label classifications this array is two dimensional.
                  The first dimension of the array should be equal to the dimension of X.
        """
        if y.ndim == 1:
            y = y[:, np.newaxis]

        word_count_matrix = self.count_vectorizer.fit_transform(X)
        self.p_c = [None] * y.shape[1]  # P(class)
        self.p_w_given_c = [None] * y.shape[1]  # P(word_i | class)
        self.p_c_given_w = [None] * y.shape[1]  # P(class | word_i)
        for label in range(y.shape[1]):
            num_classes = np.max(y[:, label]) + 1
            self.p_w_given_c[label] = np.zeros((word_count_matrix.shape[1], num_classes))
            self.p_c[label] = np.zeros(num_classes)
            for class_ in range(num_classes):
                sample_indices = np.where(y[:, label] == class_)[0]
                self.p_c[label][class_] = sample_indices.shape[0] / y.shape[0]
                self.p_w_given_c[label][:, class_] = (word_count_matrix[sample_indices, :].sum(
                    axis=0) + 1) / (word_count_matrix[sample_indices, :].sum() + word_count_matrix.shape[1])
            p_w_and_c = self.p_w_given_c[label] * self.p_c[label]  # P(w and class)
            p_w = p_w_and_c.sum(axis=1)  # P(w)
            p_w_extended = np.repeat(p_w[:, np.newaxis], self.p_w_given_c[label].shape[1], axis=1)
            self.p_c_given_w[label] = p_w_and_c / p_w_extended  # P(C | w)

    def predict(self, X):
        """ Predicts the classes for the input texts X.

        :param X: An array of texts to predict the classes to.
        :return: A two dimensional array of predicted classes.
                    First dimension is the label (for multi-class classification).
                    Second dimension is the corresponding text index of the input X.
        """
        X_bow = [self.bow_vector(x) for x in X]
        predictions = np.zeros((len(X_bow), len(self.p_c)))
        for x_index in range(len(X_bow)):
            for label_index in range(len(self.p_c)):
                prod_conditionals = np.ma.log(self.p_w_given_c[0].T * X_bow[x_index].T).filled(0).sum(axis=1) # prod P(w | class)
                weighted_by_class = np.log(self.p_c[label_index]) + prod_conditionals
                predictions[x_index][label_index] = weighted_by_class.argmax()
        return predictions.astype('int')

    def explain(self, x, method='probability', label=0, class_to_explain=None, options=None, domain_mapping=True):
        """ Generates an attribution based explanation for SVM text categorizations.

        :param x: The text to classify and generate an explanation for.
        :param method: he explainability method to use. Only 'probability' available. Default: 'probability'.
        :param label: The label to explain.
        :param class_to_explain: The numeric class label to explain.
        :param options: Ignored parameter. No options for Naive Bayes explanations.
        :param domain_mapping: Whether the relevance scores should be mapped onto the input text (True)
                               or the BoW representation (False).
        :return: A list of dictionaries that represent the tokens of the texts with the assigned relevance scores,
                 if domain_mapping is true.
                 Otherwise exports the relevance scores in a vector with the same dimension of the BoW representation.
        """
        if method == 'probability':
            explanation = self._explain(x, label=label, class_to_explain=class_to_explain)
        else:
            raise ValueError('Explainability method not implemented')
        if domain_mapping:
            return self._map_explanation_to_text(explanation[0], x), explanation[1]
        else:
            return explanation

    def _map_explanation_to_text(self, explanation, text):
        """Maps relevance scores from the given explanation in array representation
           to a list of dictionaries that represent the tokens of the texts.
        :param text: The text to map the relevance scores on.
        :return: A list of dictionaries that represent the tokens of the texts with assigned relevance scores.
        """
        preprocessed_text = self.preprocess_text(text, significant_only=False)

        voc = self.count_vectorizer.vocabulary_
        for token in preprocessed_text:
            if 'word' in token and ('ignore' not in token or not token['ignore']):
                token['relevance'] = explanation[voc[token['word']]]
        return preprocessed_text

    def _explain(self, x, label=0, class_to_explain=None):
        """ Assigns a relevance of P(c|w) for each word that is not zero in the BoW representation.

        :param x: The text to explain.
        :param label: The label to explain.
        :param class_to_explain: The class to explain.
        :return: An array with the same dimension as the BoW representation, that contains the relevance scores.
        """
        y = self.predict([x])[0][label].item()

        if class_to_explain is None:
            class_to_explain = y

        x_bow = self.bow_vector(x)
        x_binary_bow = (x_bow > 0).astype('int')
        explanation = self.p_c_given_w[label][:, class_to_explain] * x_binary_bow

        return explanation, y

    def export_model(self, model):
        """ Exports the model to the given directory.

            A model is exported to a directory that contains the model ('classifier')
            and the parameters ('parameters.pkl') as files.

        :param model: The directory to export the model to.
        """
        os.mkdir(model)
        parameter_path = os.path.join(model, 'parameters.pkl')
        parameters = {'stem': self.stem, 'tokenize': self.tokenize, 'label_mapping': self.label_mapping,
                      'word_regex': self.word_regex, 'ignored_regexes': self.ignored_regexes,
                      'p_C': self.p_c, 'p_w_given_C': self.p_w_given_c, 'p_C_given_w': self.p_c_given_w,
                      'count_vectorizer': self.count_vectorizer}
        with open(parameter_path, 'wb') as f:
            dill.dump(parameters, f)

    @classmethod
    def import_model(cls, model, *kargs, **kwargs):
        """ Imports a model from the given directory.

        :param model: The directory to the exported model.
        :return: The imported model.
        """
        parameter_path = os.path.join(model, 'parameters.pkl')
        with open(parameter_path, 'rb') as f:
            parameters = dill.load(f)
        new_model = cls(tokenize=parameters['tokenize'], label_mapping=parameters['label_mapping'],
                        ignored_regexes=parameters['ignored_regexes'], stem=parameters['stem'],
                        word_regex=parameters['word_regex'])
        new_model.p_c = parameters['p_C']
        new_model.p_w_given_c = parameters['p_w_given_C']
        new_model.p_c_given_w = parameters['p_C_given_w']
        new_model.count_vectorizer = parameters['count_vectorizer']
        return new_model

    def preprocess_text(self, text, significant_only=True):
        """ Preprocesses the given text.

        :param text: The text to preprocess.
        :param significant_only: Whether the preprocessed text should be returned as list of not ignored tokens
                                 in string format or as a list of all (ignored and not ignored) tokens
                                 in dictionary format. Ignored tokens have the attribute ignore set to True.
        :return: A list of not ignored tokens in string format, if significant_only is true.
                 Otherwise a list of all (ignored and not ignored) tokens in dictionary format.
        """
        if significant_only:
            for regex in self.ignored_regexes:
                text = re.sub(regex, ' ', text)
                regex = '(' + regex + ')'
            text = re.sub(r' ', ' spacechar ', text)
            text = re.sub(r'\n', '\nnewlinechar\n', text)
            tokens = self.tokenize(text)
            return [self.stem(token) for token in tokens if re.fullmatch('(?!spacechar|newlinechar)' + self.word_regex, token)]
        else:
            text = re.sub(r' ', ' spacechar ', text)
            text = re.sub(r'\n', '\nnewlinechar\n', text)
            return self._recursive_preprocess_text(text, self.ignored_regexes.copy())

    def _recursive_preprocess_text(self, text, ignored_regex_list):
        """ A helper function to preprocess text recursively and filter out ignored regexes.

        :param text: The text to preprocess.
        :param ignored_regex_list: A list of regexes to filter out.
        :return: A list of all (ignored and not ignored) tokens in dictionary format.
                 Ignored tokens have the attribute ignore set to True.
        """
        preprocessed_text = []
        if ignored_regex_list:
            regex = '(' + ignored_regex_list[0] + ')'
            del ignored_regex_list[0]
            preprocessed_text = re.split(regex, text)
            iter = reversed(range(len(preprocessed_text)))
            for i in iter:
                if type(preprocessed_text[i]) == str:
                    if type(preprocessed_text[i]) == str and re.fullmatch(regex, preprocessed_text[i]):
                        preprocessed_text[i] = {'token': preprocessed_text[i], 'ignore': True}
                    else:
                        tokens = self._recursive_preprocess_text(preprocessed_text[i], ignored_regex_list)
                        del preprocessed_text[i]
                        for t in reversed(tokens):
                            preprocessed_text.insert(i, t)
        else:
            voc = self.count_vectorizer.get_feature_names()
            tokens = self.tokenize(text)
            for t in tokens:
                if t == 'spacechar':
                    obj = {'token': ' ', 'ignore': True}
                elif t == 'newlinechar':
                    obj = {'token': '\n', 'ignore': True}
                else:
                    obj = {'token': t}
                    if not re.fullmatch(self.word_regex, t.lower()):
                        obj['ignore'] = True
                    elif not self.stem(t.lower()) in voc:
                        obj['ignore'] = True
                    else:
                        obj['word'] = self.stem(t.lower())
                preprocessed_text.append(obj)
        return preprocessed_text

    def bow_vector(self, x):
        bow = self.count_vectorizer.transform([x]).toarray()[0]
        return bow

