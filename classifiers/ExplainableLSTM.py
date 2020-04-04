import numpy as np
import sklearn
import re
import nltk
import dill
import os
import lime.lime_base as lime
from keras.models import Model, load_model
from keras.layers import LSTM, Dense, Bidirectional, Input, Activation
from keras.utils import to_categorical
from keras.optimizers import Adam

from LRP_for_LSTM.LSTM_bidi import LSTM_bidi
from ExplainableClassifier import ExplainableClassifier


class ExplainableLSTM(ExplainableClassifier):
    """ A wrapper class around a Keras LSTM
        that enables attribution-based explanations for text categorizations."""

    def __init__(self, wordvectors, tokenize=nltk.tokenize.word_tokenize, label_mapping=None,
                 ignored_regexes=None, max_input_size=150, lstm_units=32, word_regex=r'.*?'):
        """ Constructor for ExplainableLSTM.

        :param wordvectors: The word vectors to use in a dictionary form with words as keys and vectors as values.
        :param tokenize: The tokenize function the LSTM uses for preprocessing.
        :param label_mapping: The class labels in the form of a two dimensional array.
                              Labels should be reached by: label_mapping[label_number, class_number].
        :param ignored_regexes: A list of regexes that the LSTM should ignore for text categorizations.

        :param max_input_size: The maximum number of tokens to process with the LSTM.
        :param lstm_units: The hidden layer dimension.
        :param word_regex: A regex that a token must match to be considered by the LSTM.

        """
        self.wordvectors = wordvectors
        self.word_regex = word_regex
        self.max_input_size = max_input_size
        self.lstm_units = lstm_units
        self.wordvector_dimension = wordvectors['the'].shape[0]
        self.keras_model = None
        self.lrp_explainer = None
        super().__init__(tokenize=tokenize, label_mapping=label_mapping, ignored_regexes=ignored_regexes)

    def train(self, X, y, batch_size=32, epochs=3, learning_rate=0.001):
        """ Trains the LSTM on the the given dataset X with the given labels y.

        :param X: An array of texts the LSTM should be trained on.
        :param y: An array of numeric class labels for X. For multi-label classifications this array is two dimensional.
                  The first dimension of the array should be equal to the dimension of X.
        :param batch_size: The batch size to pass to Keras fit method.
        :param epochs: The number of epochs to train the LSTM for.
        :param learning_rate: The learning rate for LSTM training.
        """
        # Allocate numpy array for training data input
        X_preprocessed = np.zeros((len(X), self.max_input_size, self.wordvector_dimension))

        # Convert texts to word embedding representation
        for i, x in enumerate(X):
            X_preprocessed[i, :, :] = self._word_vectors(self.preprocess_text(x))[0]

        # Generalize the dimensionality of y
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # Create generic label_mapping if it is None
        if self.label_mapping is None:
            self.label_mapping = [None] * y.shape[1]

        # Input layer of model
        input_ = Input(shape=(self.max_input_size, self.wordvector_dimension), name='input')
        # LSTM layer of model
        lstm_out = Bidirectional(LSTM(self.lstm_units, recurrent_activation='sigmoid'), name='lstm')(input_)

        dense_layers = [None] * y.shape[1]
        activation_layers = [None] * y.shape[1]
        # Define losses and labels in a dict representation used by Keras
        losses = {}
        y_labeled = {}

        for label in range(y.shape[1]):
            # Fill label_mapping for each label, if it is empty
            if self.label_mapping[label] is None:
                self.label_mapping[label] = list(range(max(y[:, label]) + 1))
            y_labeled[str(label) + '_label'] = y[:, label]
            num_categories = len(set(y_labeled[str(label) + '_label']))
            # If there are more than two classes convert labels to categorial and use softmax activation and 
            # categorial_corssentropy as loss 
            if num_categories > 2:
                y_labeled[str(label) + '_label'] = to_categorical(y_labeled[str(label) + '_label'])
                dense_layers[label] = Dense(num_categories, use_bias=False, name='dense_' + str(label))(lstm_out)
                activation_layers[label] = Activation('softmax', name=str(label) + '_label')(dense_layers[label])
                losses[str(label) + '_label'] = 'categorical_crossentropy'
            # If there are more than two classes use sigmoid activation and binary_crossentropy as loss
            else:
                dense_layers[label] = Dense(1, use_bias=False, name='dense_' + str(label))(lstm_out)
                activation_layers[label] = Activation('sigmoid', name=str(label) + '_label')(dense_layers[label])
                losses[str(label) + '_label'] = 'binary_crossentropy'

        # Compile and train the LSTM
        self.keras_model = Model(inputs=input_, outputs=activation_layers)
        opt = Adam(lr=learning_rate)
        self.keras_model.compile(optimizer=opt, loss=losses, metrics=["accuracy"])
        self.keras_model.fit(X_preprocessed, y_labeled, batch_size=batch_size, epochs=epochs)

        # Build the LSTM model by Arras (LSTM_bidi) to perform LRP on, this sets self.lrp_explainer
        self._build_lrp_lstm()

    def predict_proba(self, X):
        """ Predicts the probabilities of classes for the given texts X.

        :param X: An array of texts to predict the probabilities to.
        :return: A three dimensional array of probabilities.
                    First dimension is the label (for multi-class classification).
                    Second dimension is the corresponding text index of the input X.
                    Third dimension is the class number.
        """

        X_preprocessed = np.array([self._word_vectors(self.preprocess_text(x))[0] for x in X])
        y = self.keras_model.predict(X_preprocessed)

        # Generalize the dimensionality of y, in case there is only one label
        if type(y) != list:
            y = [y]
        return y

    def predict(self, X):
        """ Predicts the classes for the input texts X.

        :param X: An array of texts to predict the classes to.
        :return: A two dimensional array of predicted classes.
                 First dimension is the label (for multi-class classification).
                 Second dimension is the corresponding text index of the input X.
        """
        y = self.predict_proba(X)

        # Convert probabilities to the the maximum class for each label
        y_formatted = np.zeros((len(X), len(y)), dtype='int')
        for idx, label in enumerate(y):
            if label.shape[1] == 1:
                y_formatted[:, idx] = label.round().flatten().astype('int')
            else:
                y_formatted[:, idx] = np.argmax(label, axis=1)
        return y_formatted

    def explain(self, x, method='lrp', label=0, class_to_explain=None, options=None, domain_mapping=True):
        """ Generates an attribution based explanation for LSTM text categorizations.

        :param x: The text to classify and generate an explanation for.
        :param method: The explainability method to use. Options: 'lrp' and 'lime'. Default: 'lrp'.
        :param label: The label to explain.
        :param class_to_explain: The numeric class label to explain.
        :param options: Options for LRP or LIME explanations.
        :param domain_mapping: Whether the relevance scores should be mapped onto the input text (True)
                               or the word vector representation (False).
        :return: A list of dictionaries that represent the tokens of the texts with the assigned relevance scores
                 and the predicted class, if domain_mapping is true.
                 Otherwise exports the relevance scores in a vector with the same dimension as the word vector
                 representation and the predicted class.
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
                      'ignored_regexes': self.ignored_regexes, 'max_input_size': self.max_input_size,
                      'lstm_units': self.lstm_units}

        self.keras_model.save(classifier_path)
        with open(parameters_path, 'wb') as f:
            dill.dump(parameters, f)

    @classmethod
    def import_model(cls, model, wordvectors, *kargs, **kwargs):
        """ Imports a model from the given directory.

        :param model: The directory to the exported model.
        :param wordvectors: The filepath to the GloVe word vector file to load.
        :return: The imported model.
        """
        classifier_path = os.path.join(model, 'classifier')
        parameters_path = os.path.join(model, 'parameters.pkl')

        with open(parameters_path, 'rb') as f:
            parameters = dill.load(f)

        new_model = cls(wordvectors, tokenize=parameters['tokenize'], label_mapping=parameters['label_mapping'],
                        ignored_regexes=parameters['ignored_regexes'], max_input_size=parameters['max_input_size'],
                        lstm_units=parameters['lstm_units'], word_regex=parameters['word_regex'])
        new_model.keras_model = load_model(classifier_path)
        new_model._build_lrp_lstm()

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
            return [token for token in tokens if re.fullmatch('(?!spacechar|newlinechar)' + self.word_regex, token)]
        else:
            text = re.sub(r' ', ' spacechar ', text)
            text = re.sub(r'\n', '\nnewlinechar\n', text)
            return self._recursive_preprocess_text(text, self.ignored_regexes.copy())

    @staticmethod
    def load_glove_wordvectors(filepath):
        """ Loads word vectors form a GloVe word vector file.

        :param filepath: The filepath to the GloVe word vector file.
        :return: A dictionary that contains the word vectors.
        """
        embeddings = {}
        print("Load word embeddings")
        with open(filepath, 'r', encoding='utf8') as f:
            for line in f:
                splitted_line = line.split()
                word = splitted_line[0]
                vector = np.array([float(v) for v in splitted_line[1:]])
                embeddings[word] = vector
        print("Word embeddings loaded.")
        return embeddings

    def _recursive_preprocess_text(self, text, ignored_regex_list):
        """ A helper function to preprocess text recursively and filter out ignored regexes.

        :param text: The text to preprocess.
        :param ignored_regex_list: A list of regexes to filter out.
        :return: A list of all (ignored and not ignored) tokens in dictionary format.
                 Ignored tokens have the attribute ignore set to True.
        """
        preprocessed_text = []
        # Ignore all regexes that are in the ignored_regex_list
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
        # Tokenize Text and assign information to each token
        else:
            tokens = self.tokenize(text)
            counter = 0
            for t in tokens:
                lower_t = t.lower()
                if t == 'spacechar':
                    obj = {'token': ' ', 'ignore': True}
                elif t == 'newlinechar':
                    obj = {'token': '\n', 'ignore': True}
                else:
                    obj = {'token': t}
                    if counter >= self.max_input_size:
                        obj['ignore'] = True
                    elif not re.fullmatch(self.word_regex, t.lower()):
                        obj['ignore'] = True
                    elif not lower_t in self.wordvectors:
                        obj['ignore'] = True
                    else:
                        counter += 1
                preprocessed_text.append(obj)
        return preprocessed_text

    def _word_vectors(self, tokenized_text):
        """ Converts a tokenized text to an array of word vectors.

        :param tokenized_text: The tokenized text to convert to word vectors.
        :return: an array of word vectors.
        """
        text_word_vectors = np.zeros((self.max_input_size, self.wordvector_dimension))
        counter = 0
        # If a list of only relevant tokens is given
        if tokenized_text and type(tokenized_text[0]) is str:
            for token in tokenized_text:
                if counter >= self.max_input_size:
                    break
                if token.lower() in self.wordvectors:
                    text_word_vectors[counter, :] = self.wordvectors[token.lower()]
                    counter += 1
        # If a list of ignored and relevant tokens is given
        else:
            for token in tokenized_text:
                if counter >= self.max_input_size:
                    break
                if not ('ignore' in token and token['ignore']):
                    text_word_vectors[counter, :] = self.wordvectors[token['token'].lower()]
                    counter += 1
                else:
                    token['ignore'] = True

        return text_word_vectors, tokenized_text

    def _build_lrp_lstm(self):
        """ Creates Arras' LRP_for_LSTM model (https://github.com/ArrasL/LRP_for_LSTM)
        from the trained Keras LSTM model.
        """
        # LSTM Keras weights and weights in Arras' LSTM_bidi implementation are not ordered in the same way
        def reorder_weights(weights):
            wi = weights[:self.lstm_units]
            wf = weights[self.lstm_units:2 * self.lstm_units]
            wg = weights[2 * self.lstm_units:3 * self.lstm_units]
            wo = weights[3 * self.lstm_units:]
            return np.vstack((wi, wg, wf, wo))

        lstm_weights = self.keras_model.get_layer('lstm').get_weights()
        model_weights = {'Wxh_Left': reorder_weights(lstm_weights[0].T),
                         'bh_Left': reorder_weights(lstm_weights[2].T).flatten(),
                         'Whh_Left': reorder_weights(lstm_weights[1].T),
                         'Wxh_Right': reorder_weights(lstm_weights[3].T),
                         'bh_Right': reorder_weights(lstm_weights[5].T).flatten(),
                         'Whh_Right': reorder_weights(lstm_weights[4].T)}

        # Build a LSTM_bidi object for each label
        self.lrp_explainer = [None] * len(self.keras_model.get_config()['output_layers'])
        for dense_index in range(len(self.keras_model.get_config()['output_layers'])):
            dense_weights = self.keras_model.get_layer('dense_' + str(dense_index)).get_weights()
            model_weights['Why_Left'] = dense_weights[0][:self.lstm_units, :].T
            model_weights['Why_Right'] = dense_weights[0][self.lstm_units:, :].T
            self.lrp_explainer[dense_index] = LSTM_bidi(model_weights.copy())

    def _explain_instance_lrp(self, x, label=0, class_to_explain=None,
                              eps=0.001, bias_factor=1.0, domain_mapping=True, **kwargs):
        """ Explains a text classification with LRP.

        :param x: The text to classify and generate an explanation for.
        :param label: The label to explain.
        :param class_to_explain: The numeric class label to explain.
        :param eps: The stabilization term epsilon.
        :param bias_factor: The bias factor.
                            If it is equal to one the bias term will be split equally onto
                            relevance scores of all words.
                            If it is zero the bias term will not be included into the relevance scores.
        param domain_mapping: Whether the relevance scores should be mapped onto the input text (True)
                               or the word vector representation (False).
        :return: A list of dictionaries that represent the tokens of the texts with the assigned relevance scores
                 and the predicted class, if domain_mapping is true.
                 Otherwise exports the relevance scores in a vector with the same dimension of the word vector
                 representation and the predicted class.
        """
        # Get word vector representation of text and the tokenized text
        x, tokenized_text = self._word_vectors(self.preprocess_text(x, significant_only=False))

        # Predict with lrp_explainer (LSTM_bidi)
        self.lrp_explainer[label].set_input(x)
        scores = self.lrp_explainer[label].forward()

        # For two classes ...
        if scores.shape[0] == 1:
            y = np.sign(scores).astype('int').item()
            y = 0 if y == -1 else 1

            if class_to_explain is None:
                class_to_explain = y
            # Explain the outcome for class 1 with LRP
            relevance_left, relevance_right, _ = self.lrp_explainer[label].lrp(x, 0, eps=eps, bias_factor=bias_factor)
        # ... for more than two classes
        else:
            y = scores.argmax().item()
            # Set class_to_explain to predicted class if not given
            if class_to_explain is None:
                class_to_explain = y

            # Explain the outcome for class_to_explain with LRP
            relevance_left, relevance_right, _ = self.lrp_explainer[label].lrp(x, class_to_explain, eps=eps,
                                                                               bias_factor=bias_factor)

        # Add up relevances for both directions in the bidirectional LSTM
        relevance = relevance_left + relevance_right
        # Reverse explanation weights, if class_to_explain is 0
        if class_to_explain == 0:
            relevance = -relevance

        relevance = relevance.sum(axis=1)

        if domain_mapping:
            counter = 0
            for token in tokenized_text:
                if counter >= self.max_input_size:
                    break
                if 'ignore' in token and token['ignore']:
                    continue
                token['relevance'] = relevance[counter]
                counter += 1
            return tokenized_text, y
        else:
            return relevance, y

    def _explain_instance_lime(self, x, label=0, class_to_explain=None, sample_size=1000, kernel_width=10,
                               distance_metric='cosine', domain_mapping=True, **kwargs):
        """ Explains a text classification with LIME.

        :param x: The text to classify and generate an explanation for.
        :param label: The label to explain.
        :param class_to_explain: The numeric class label to explain.

        :param sample_size: The number of samples the interpretable LIME surrogate model should be trained on.
        :param kernel_width: The width of the kernel that specifies the proximity of a perturbed text to the original
                             text.
        :param distance_metric: The distance metric to use for the proximity of a perturbed text to the original
                                text. Default: 'cosine'.
        :param domain_mapping: Whether the relevance scores should be mapped onto the input text (True)
                               or the word vector representation (False).
        :return: A list of dictionaries that represent the tokens of the texts with the assigned relevance scores,
                 if domain_mapping is true.
                 Otherwise exports the relevance scores in a vector with the same dimension of the word vector
                 representation.
        """
        # Returns the distances of each row vector compared to the first row vector in the matrix
        def distance_function(x):
            return sklearn.metrics.pairwise.pairwise_distances(x, [x[0]], metric=distance_metric)[:, 0] * 100

        # Defines the kernel function used for LIME with adjustable width
        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        x_escaped = re.sub(r' ', ' spacechar ', x)
        x_escaped = re.sub(r'\n', '\nnewlinechar\n', x_escaped)
        tokens_with_space_and_newline = self.tokenize(x_escaped)
        tokens = list(filter(lambda t: t != 'spacechar' and t != 'newlinechar', tokens_with_space_and_newline))

        y = self.predict([x])[0][label]

        # If there is no relevant word in x
        if len(tokens) == 0:
            if domain_mapping:
                return [], y.item()
            else:
                return np.zeros(self.max_input_size), y.item()

        mask_vectors = np.ones((sample_size + 1, len(tokens)))
        num_of_word_deletions = np.random.randint(1, len(tokens) + 1, sample_size)

        modified_texts = [None] * (sample_size + 1)
        modified_texts[0] = ' '.join(tokens)

        for i, size in enumerate(num_of_word_deletions, start=1):
            inactive = np.random.choice(range(len(tokens)), size, replace=False)
            mask_vectors[i, inactive] = 0
            modified_texts[i] = ' '.join(np.delete(np.array(tokens), inactive))

        distances = distance_function(mask_vectors)
        predictions = self.predict_proba(modified_texts)[label]

        # Get relevances with the LimeBase
        lime_explainer = lime.LimeBase(kernel)
        if predictions.shape[1] == 1:
            if class_to_explain is None:
                class_to_explain = y
            lime_explanation = lime_explainer.explain_instance_with_data(mask_vectors, predictions, distances, 0,
                                                                         len(tokens),
                                                                         feature_selection='none')[1]
            if class_to_explain == 0:
                lime_explanation = [(i, -r) for i, r in lime_explanation]
        else:
            if class_to_explain is None:
                class_to_explain = y
            lime_explanation = lime_explainer.explain_instance_with_data(mask_vectors, predictions, distances,
                                                                         class_to_explain,
                                                                         len(tokens),
                                                                         feature_selection='none')[1]

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
            counter = 0
            explanation = np.zeros(self.max_input_size)
            for index, token in enumerate(tokens):
                if re.fullmatch(self.word_regex, token.lower()) \
                        and token.lower() in self.wordvectors \
                        and index in lime_explanation:
                    explanation[counter] = lime_explanation[index]
                    counter += 1
                    if counter >= self.max_input_size:
                        break

        return explanation, y.item()
