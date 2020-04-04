import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import dill
import os
import lime.lime_base as lime

from ExplainableClassifier import ExplainableClassifier


class ExplainableSVM(ExplainableClassifier):
    """ A wrapper class around sklearn.svm.LinearSVC
        that enables attribution-based explanations for text categoizations.
    """

    def __init__(self, tokenize=nltk.tokenize.word_tokenize, label_mapping=None,
                 ignored_regexes=None, word_regex=r'[\'\w]+',
                 stem=nltk.stem.snowball.EnglishStemmer().stem):
        """ Constructor for ExplainableSVM.

        :param tokenize: The tokenize function the SVM uses for preprocessing.
        :param label_mapping: The class labels in the form of a two dimensional array.
                              Labels should be reached by: label_mapping[label_number, class_number].
        :param ignored_regexes: A list of regexes that the SVM should ignore for text categorizations.
        :param word_regex: A regex that a token must match to be considered by the SVM.
        :param stem: The stemming function the SVM uses for preprocessing of the text.
        """
        self.word_regex = word_regex
        self.stem = stem
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.preprocess_text, analyzer='word',
                                                token_pattern='(?!spacechar|newlinechar)' + self.word_regex)
        self.clf = None
        super().__init__(tokenize=tokenize, label_mapping=label_mapping,
                         ignored_regexes=ignored_regexes)

    def train(self, X, y, max_iter=10000, C=1):
        """ Trains the SVM on the the given dataset X with the given labels y.

        :param X: An array of texts the SVM should be trained on.
        :param y: An array of numeric class labels for X. For multi-label classifications this array is two dimensional.
                  The first dimension of the array should be equal to the dimension of X.
        :param max_iter: The maximum number of iterations to be run.
        :param C: Penalty parameter C of the error term.
        """
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # Create TF-IDF Vectors
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.preprocess_text, analyzer='word',
                                                token_pattern='(?!spacechar|newlinechar)' + self.word_regex)
        X_tfidf = self.tfidf_vectorizer.fit_transform(X)

        if self.label_mapping is None:
            self.label_mapping = [None] * y.shape[1]

        self.clf = [None] * y.shape[1]
        for label in range(y.shape[1]):
            self.clf[label] = sklearn.svm.LinearSVC(max_iter=max_iter, C=C)
            self.clf[label].fit(X_tfidf, y[:, label])

            if self.label_mapping[label] is None:
                self.label_mapping[label] = list(range(min(y[:, label]), max(y[:, label]) + 1))

    def predict_proba(self, X, softmax_base=np.e):
        """ Predicts the probabilities of classes for the given texts X by normalizing SVM scores with a softmax.

        :param X: An array of texts to predict the probabilities to.
        :param softmax_base: The base to use for the softmax. Default will be np.e.
        :return: A three dimensional array of probabilities.
                    First dimension is the label (for multi-class classification).
                    Second dimension is the corresponding text index of the input X.
                    Third dimension is the class number.
        """
        X_tfidf = self.tfidf_vectorizer.transform(X)

        def softmax(x, b):
            return (np.power(b, x).T / np.sum(np.power(b, x), axis=1)).T

        probabilities = [None] * len(self.clf)
        for i in range(len(self.clf)):
            decision = self.clf[i].decision_function(X_tfidf)
            is_binary = self.clf[i].classes_.shape[0] == 2
            if is_binary:
                prob_matrix = softmax(np.array([1 - decision, decision]).T, softmax_base)
            else:
                prob_matrix = softmax(decision, softmax_base)
            probabilities[i] = prob_matrix
        return probabilities

    def predict(self, X):
        """ Predicts the classes for the input texts X.

        :param X: An array of texts to predict the classes to.
        :return: A two dimensional array of predicted classes.
                    First dimension is the label (for multi-class classification).
                    Second dimension is the corresponding text index of the input X.
        """
        X_tfidf = self.tfidf_vectorizer.transform(X)
        prediction = [0] * len(self.clf)
        for label in range(len(self.clf)):
            prediction[label] = self.clf[label].predict(X_tfidf)

        return np.array(prediction).T

    def explain(self, x, method='lrp', label=0, class_to_explain=None, options=None, domain_mapping=True):
        """ Generates an attribution based explanation for SVM text categorizations.

        :param x: The text to classify and generate an explanation for.
        :param method: The explainability method to use. Options: 'lrp' and 'lime'. Default: 'lrp'.
        :param label: The label to explain.
        :param class_to_explain: The numeric class label to explain.
        :param options: Options for LRP or LIME explanations.
        :param domain_mapping: Whether the relevance scores should be mapped onto the input text (True)
                               or the TFIDF representation (False).
        :return: A list of dictionaries that represent the tokens of the texts with the assigned relevance scores,
                 if domain_mapping is true.
                 Otherwise exports the relevance scores in a vector with the same dimension of the TFIDF representation.
        """
        if options is None:
            options = {}

        options['domain_mapping'] = domain_mapping
        if method == 'lrp':
            explanation = self._explain_instance_lrp(x, label=label, class_to_explain=class_to_explain, **options)
        elif method == 'lime':
            explanation = self._explain_instance_lime(x, label=label, class_to_explain=class_to_explain, **options)
        else:
            raise ValueError('Explainability method not implemented')

        return explanation

    def export_model(self, model):
        """ Exports the model to the given directory.

            A model is exported to a directory that contains the model ('classifier')
            and the parameters ('parameters.pkl') as files.

        :param model: The directory to export the model to.
        """
        os.mkdir(model)
        classifier_path = os.path.join(model, 'classifier')
        parameters_path = os.path.join(model, 'parameters.pkl')
        parameters = {'label_mapping': self.label_mapping, 'word_regex': self.word_regex, 'tokenize': self.tokenize,
                      'ignored_regexes': self.ignored_regexes, 'stem': self.stem,
                      'tfidf_vectorizer': self.tfidf_vectorizer}
        with open(classifier_path, 'wb') as f:
            dill.dump(self.clf, f)
        with open(parameters_path, 'wb') as f:
            dill.dump(parameters, f)

    @classmethod
    def import_model(cls, model):
        """ Imports a model from the given directory.

        :param model: The directory to the exported model.
        :return: The imported model.
        """
        classifier_path = os.path.join(model, 'classifier')
        parameters_path = os.path.join(model, 'parameters.pkl')
        with open(parameters_path, 'rb') as f:
            parameters = dill.load(f)
        with open(classifier_path, 'rb') as f:
            classifier = dill.load(f)
        new_model = cls(tokenize=parameters['tokenize'], label_mapping=parameters['label_mapping'],
                        ignored_regexes=parameters['ignored_regexes'], word_regex=parameters['word_regex'],
                        stem=parameters['stem'])
        new_model.tfidf_vectorizer = parameters['tfidf_vectorizer']
        new_model.clf = classifier
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
                regex = '(' + regex + ')'
                text = re.sub(regex, ' ', text)
            text = re.sub(r' ', ' spacechar ', text)
            text = re.sub(r'\n', '\nnewlinechar\n', text)
            tokens = self.tokenize(text)
            return [self.stem(token) for token in tokens if
                    re.fullmatch('(?!spacechar|newlinechar)' + self.word_regex, token)]
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
            reversed_range = reversed(range(len(preprocessed_text)))
            for i in reversed_range:
                if type(preprocessed_text[i]) == str:
                    if type(preprocessed_text[i]) == str and re.fullmatch(regex, preprocessed_text[i]):
                        preprocessed_text[i] = {'token': preprocessed_text[i], 'ignore': True}
                    else:
                        tokens = self._recursive_preprocess_text(preprocessed_text[i], ignored_regex_list)
                        del preprocessed_text[i]
                        for t in reversed(tokens):
                            preprocessed_text.insert(i, t)
        else:
            voc = self.tfidf_vectorizer.get_feature_names()
            tokens = self.tokenize(text)
            for t in tokens:
                if t == 'spacechar':
                    obj = {'token': ' ', 'ignore': True}
                elif t == 'newlinechar':
                    obj = {'token': '\n', 'ignore': True}
                elif not re.fullmatch(self.word_regex, t.lower()):
                    obj = {'token': t, 'ignore': True}
                elif not self.stem(t.lower()) in voc:
                    obj = {'token': t, 'ignore': True}
                else:
                    obj = {'token': t}
                preprocessed_text.append(obj)
        return preprocessed_text

    def _explain_instance_lrp(self, x, label=0, class_to_explain=None, eps=0.001, bias_factor=1.0, domain_mapping=True,
                              split_bow=True, **kwargs):
        x_tfidf = self.tfidf_vectorizer.transform([x]).toarray()[0]

        is_binary = self.clf[label].classes_.shape[0] == 2
        y = self.predict([x])[0][label]
        svm_scores = self.clf[label].decision_function([x_tfidf])[0]
        negative_score = False

        if is_binary:
            if class_to_explain is None:
                class_to_explain = y
            score_to_explain = svm_scores.item()
            if class_to_explain == 0:
                negative_score = True

        else:
            if class_to_explain is None:
                class_to_explain = y
                score_to_explain = svm_scores[class_to_explain]
            else:
                score_to_explain = svm_scores[class_to_explain]

        explanation = np.zeros_like(x_tfidf)
        word_indices = x_tfidf.nonzero()[0]

        for word_id in word_indices:
            if is_binary:
                relevance = self.clf[label].coef_[0][word_id] \
                        * x_tfidf[word_id] \
                        + (bias_factor * (self.clf[label].intercept_[0]) / len(word_indices))
                relevance = relevance * np.abs(score_to_explain)
                if negative_score:
                    relevance = -relevance
            else:
                sign_out = np.sign(score_to_explain)
                relevance = self.clf[label].coef_[class_to_explain][word_id] * x_tfidf[word_id] + (
                        bias_factor * (self.clf[label].intercept_[class_to_explain] + eps * sign_out) / len(
                    word_indices))
                relevance = relevance * np.abs(score_to_explain)
            explanation[word_id] = relevance

        tokenized_text = self.preprocess_text(x)
        if domain_mapping:
            voc = self.tfidf_vectorizer.vocabulary_
            expl = self.preprocess_text(x, significant_only=False)
            for token in expl:
                if 'ignore' in token:
                    continue
                if 'ignore' not in token or not token['ignore']:
                    token_stem = self.stem(token['token'].lower())
                    if split_bow:
                        token['relevance'] = explanation[voc[token_stem]] / tokenized_text.count(token_stem)
                    else:
                        token['relevance'] = explanation[voc[token_stem]]
            explanation = expl

        return explanation, y.item()

    def _explain_instance_lime(self, x, label=0, class_to_explain=None, sample_size=1000, kernel_width=10,
                               distance_metric='cosine', softmax_base=np.e, domain_mapping=True, **kwargs):
        def distance_function(x):
            return sklearn.metrics.pairwise.pairwise_distances(x, [x[0]], metric=distance_metric)[:, 0] * 100

        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        x_escaped = re.sub(r' ', ' spacechar ', x)
        x_escaped = re.sub(r'\n', '\nnewlinechar\n', x_escaped)
        tokens_with_space_and_newline = self.tokenize(x_escaped)
        tokens = list(filter(lambda t: t != 'spacechar' and t != 'newlinechar', tokens_with_space_and_newline))

        y = self.predict([x])[0][label]
        if class_to_explain is None:
            class_to_explain = y

        # If there is no relevant word in x
        if len(tokens) == 0:
            if domain_mapping:
                return [], y.item()
            else:
                return np.zeros(len(self.tfidf_vectorizer.vocabulary_)), y.item()

        mask_vectors = np.ones((sample_size + 1, len(tokens)))
        num_of_word_deletions = np.random.randint(1, len(tokens) + 1, sample_size)

        modified_texts = [None] * (sample_size + 1)
        modified_texts[0] = ' '.join(tokens)

        for i, size in enumerate(num_of_word_deletions, start=1):
            inactive = np.random.choice(range(len(tokens)), size, replace=False)
            mask_vectors[i, inactive] = 0
            modified_texts[i] = ' '.join(np.delete(np.array(tokens), inactive))

        distances = distance_function(mask_vectors)
        predictions = self.predict_proba(modified_texts, softmax_base=softmax_base)[0]

        lime_explainer = lime.LimeBase(kernel)
        lime_explanation = lime_explainer.explain_instance_with_data(mask_vectors,
                                                                     predictions, distances, class_to_explain,
                                                                     len(tokens),
                                                                     feature_selection='none')[1]
        lime_explanation.sort()
        lime_explanation = dict((index, relevance) for index, relevance in lime_explanation)

        if domain_mapping:
            explanation = []
            counter = 0
            for token in tokens_with_space_and_newline:
                if token == 'spacechar':
                    obj = {'token': ' '}
                elif token == 'newlinechar':
                    obj = {'token': '\n'}
                elif counter in lime_explanation:
                    obj = {'token': token, 'relevance': lime_explanation[counter]}
                    counter += 1
                else:
                    obj = {'token': token}
                    counter += 1

                explanation.append(obj)
        else:
            voc = self.tfidf_vectorizer.vocabulary_
            explanation = np.zeros(len(voc))
            for index, token in enumerate(tokens):
                token_stem = self.stem(token)
                if token_stem in voc and index in lime_explanation:
                    explanation[voc[token_stem]] += lime_explanation[index]
        return explanation, y.item()
