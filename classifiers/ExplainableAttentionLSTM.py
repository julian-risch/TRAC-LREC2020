import numpy as np
import re
import nltk
import dill
import os
from keras.models import Model, load_model
from keras.layers import LSTM, Dense, Bidirectional, Input, Activation
from keras.utils import to_categorical
from keras.optimizers import Adam

from AttentionLayer.AttentionLayer import AttentionLayer
from ExplainableClassifier import ExplainableClassifier


class ExplainableAttentionLSTM(ExplainableClassifier):
    """ A wrapper class around a Keras LSTM with attention mechanism
        that enables attribution-based explanations for text categorizations."""

    def __init__(self, wordvectors, tokenize=nltk.tokenize.word_tokenize, label_mapping=None,
                 ignored_regexes=None, max_input_size=150, lstm_units=32, word_regex=r'.*?'):
        """ Constructor for ExplainableAttentionLSTM.

        :param wordvectors: The word vectors to use in a dictionary form with words as keys and vectors as values.
        :param tokenize: The tokenize function the Attention LSTM uses for preprocessing.
        :param label_mapping: The class labels in the form of a two dimensional array.
                              Labels should be reached by: label_mapping[label_number, class_number].
        :param ignored_regexes: A list of regexes that the Attention LSTM should ignore for text categorizations.

        :param max_input_size: The maximum number of tokens to process with the Attention LSTM.
        :param lstm_units: The hidden layer dimension.
        :param word_regex: A regex that a token must match to be considered by the Attention LSTM.

        """
        self.wordvectors = wordvectors
        self.word_regex = word_regex
        self.max_input_size = max_input_size
        self.lstm_units = lstm_units
        self.wordvector_dimension = wordvectors['the'].shape[0]
        self.keras_model = None
        self.attention_model = None
        super().__init__(tokenize=tokenize, label_mapping=label_mapping,
                         ignored_regexes=ignored_regexes)

    def train(self, X, y, batch_size=32, epochs=3, learning_rate=0.001):
        """ Trains the Attention LSTM on the the given dataset X with the given labels y.

        :param X: An array of texts the Attention LSTM should be trained on.
        :param y: An array of numeric class labels for X. For multi-label classifications this array is two dimensional.
                  The first dimension of the array should be equal to the dimension of X.
        :param batch_size: The batch size to pass to Keras fit method.
        :param epochs: The number of epochs to train the Attention LSTM for.
        :param learning_rate: The learning rate for Attention LSTM training.
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
        lstm_out = Bidirectional(LSTM(self.lstm_units, recurrent_activation='sigmoid', return_sequences=True),
                                 name='lstm')(input_)

        attention_layers = [None] * y.shape[1]
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
                attention_layers[label] = AttentionLayer(name='attention_' + str(label))(lstm_out)
                dense_layers[label] = Dense(num_categories, use_bias=False, name='dense_' + str(label))(
                    attention_layers[label])
                activation_layers[label] = Activation('softmax', name=str(label) + '_label')(dense_layers[label])
                losses[str(label) + '_label'] = 'categorical_crossentropy'
            # If there are more than two classes use sigmoid activation and binary_crossentropy as loss
            else:
                attention_layers[label] = AttentionLayer(name='attention_' + str(label))(lstm_out)
                dense_layers[label] = Dense(1, use_bias=False, name='dense_' + str(label))(attention_layers[label])
                activation_layers[label] = Activation('sigmoid', name=str(label) + '_label')(dense_layers[label])
                losses[str(label) + '_label'] = 'binary_crossentropy'

        # Compile and train the LSTM
        self.keras_model = Model(inputs=input_, outputs=activation_layers)
        opt = Adam(lr=learning_rate)
        self.keras_model.compile(optimizer=opt, loss=losses, metrics=["accuracy"])
        self.keras_model.fit(X_preprocessed, y_labeled, batch_size=batch_size, epochs=epochs)

        # Build the LSTM model that outputs attention weights, this sets self.attention_model
        self._build_attention_model()

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

    def explain(self, x, method='attention', label=0, class_to_explain=None, options=None, domain_mapping=True):
        """ Generates an attribution based explanation for LSTM text categorizations.

        :param x: The text to classify and generate an explanation for.
        :param method: The explainability method to use. Only 'attention' available. Default: 'attention'.
        :param label: The label to explain.
        :param class_to_explain: Is not considered by explanations with attention mechanism.
        :param options: Ignored parameter. No options for attention mechanism.
        :param domain_mapping: Whether the relevance scores should be mapped onto the input text (True)
                               or the word vector representation (False).
        :return: A list of dictionaries that represent the tokens of the texts with the assigned relevance scores
                 and the predicted class, if domain_mapping is true.
                 Otherwise exports the relevance scores in a vector with the same dimension as the word vector
                 representation and the predicted class.
        """
        if method == 'attention':
            explanation = self._explain(x, label=label)
        else:
            raise ValueError('Explainability method not implemented')

        if domain_mapping:
            return self._map_explanation_to_text(explanation[0], x), explanation[1]
        else:
            return explanation

    def _explain(self, x, label=0):
        """ Returns the importance weights of the attention mechanism and the predicted class.

        :param x: The text to classify and generate an explanation for.
        :param label: The label to explain.
        :return: The importance weights of the attention mechanism and the predicted class.
        """
        y = self.predict([x])[0][label].item()
        x, tokenized_text = self._word_vectors(self.preprocess_text(x, significant_only=False))

        attention = self.attention_model.predict(np.array([x]))

        if type(attention) != list:
            attention = [attention]

        attention = attention[label][0, :, 0]

        return attention, y

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
        new_model.keras_model = load_model(classifier_path, custom_objects={'AttentionLayer': AttentionLayer})
        new_model._build_attention_model()

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
        if ignored_regex_list:
            regex = ignored_regex_list[0]
            del ignored_regex_list[0]
            preprocessed_text = re.split(regex, text)
            iter = reversed(range(len(preprocessed_text)))
            for i in iter:
                if type(preprocessed_text[i]) == str:
                    if type(preprocessed_text[i]) == str and re.fullmatch(regex, preprocessed_text[i]):
                        preprocessed_text[i] = {'token': preprocessed_text[i], 'ignore': True,
                                                'ignore-reason': 'Preprocessing Regex'}
                    else:
                        tokens = self._recursive_preprocess_text(preprocessed_text[i], ignored_regex_list)
                        del preprocessed_text[i]
                        for t in reversed(tokens):
                            preprocessed_text.insert(i, t)
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
        :return: An array of word vectors.
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

    def _map_explanation_to_text(self, explanation, text):
        """Maps relevance scores from the given explanation in array representation
           to a list of dictionaries that represent the tokens of the texts.
        :param text: The text to map the relevance scores on.
        :return: A list of dictionaries that represent the tokens of the texts with assigned relevance scores.
        """
        tokenized_text = self._word_vectors(self.preprocess_text(text, significant_only=False))[1]

        counter = 0
        for token in tokenized_text:
            if counter >= self.max_input_size:
                break
            if 'ignore' in token and token['ignore']:
                continue
            token['relevance'] = explanation[counter].item()
            counter += 1

        return tokenized_text

    def _build_attention_model(self):
        """
        Builds a Keras model that outputs the attention weights instead of predictions
        and sets it as self.attention_model.

        The attention_model is build from the weights of the keras_model.
        Make sure keras_model is set before calling this method.
        """
        input_ = Input(shape=(self.max_input_size, self.wordvector_dimension), name='input2')

        lstm_out = Bidirectional(LSTM(self.lstm_units, recurrent_activation='sigmoid', return_sequences=True),
                                 name='lstm2')(input_)
        attention_layers = []

        for label in range(len(self.label_mapping)):
            attention_layer = AttentionLayer(name='attention2_' + str(label), return_attention=True)(lstm_out)
            attention_layers.append(attention_layer)
        self.attention_model = Model(inputs=input_, outputs=attention_layers)
        # Set Weights
        old_lstm_layer = self.keras_model.get_layer('lstm')
        self.attention_model.get_layer('lstm2').set_weights(old_lstm_layer.get_weights())
        for label in range(len(self.label_mapping)):
            old_attention_layer = self.keras_model.get_layer('attention_' + str(label))
            self.attention_model.get_layer('attention2_' + str(label)).set_weights(old_attention_layer.get_weights())
