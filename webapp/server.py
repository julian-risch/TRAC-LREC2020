#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import sys
import traceback
sys.path.insert(0, '../classifiers/')
from ExplainableClassifier import ExplainableClassifier
from ExplainableLSTM import ExplainableLSTM
from ExplainableSVM import ExplainableSVM
from ExplainableNaiveBayes import ExplainableNaiveBayes
from ExplainableAttentionLSTM import ExplainableAttentionLSTM
from AttentionLayer import AttentionLayer

class HTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        rootdir = './frontend'
        try:
            if self.path == '/':
                f = open(rootdir + '/index.html', 'rb')
            elif self.path == '/label_mapping':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(self.lstm.label_mapping).encode())
                return
            else:
                f = open(rootdir + self.path, 'rb')
            self.send_response(200)

            self.send_header('Content-type', 'text-html')
            self.end_headers()

            self.wfile.write(f.read())
            f.close()
            return

        except IOError:
            self.send_error(404, 'file not found')

    def do_POST(self):

        content_len = int(self.headers.get('Content-Length'))
        input_ = json.loads(self.rfile.read(content_len).decode())

        print('Input: ', input_)

        try:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            if input_['classifier'] == 'lstm':
                explanation = self.lstm.explain(input_['text'], method=input_['method'], label=input_['label'],
                                                class_to_explain=input_['class_to_explain'],
                                                options=input_['options'])
            elif input_['classifier'] == 'svm':
                explanation = self.svm.explain(input_['text'], method=input_['method'], label=input_['label'],
                                               class_to_explain=input_['class_to_explain'],
                                               options=input_['options'])
            elif input_['classifier'] == 'naivebayes':
                explanation = self.naive.explain(input_['text'], label=input_['label'],
                                                 class_to_explain=input_['class_to_explain'])
            elif input_['classifier'] == 'att_lstm':
                explanation = self.att_lstm.explain(input_['text'], label=input_['label'])
            else:
                self.send_error(500, 'Explanation could not be generated')
                return

            self.wfile.write(json.dumps(explanation).encode())
            print('Output: ', explanation)
            return

        except:
            print(traceback.format_exc())
            self.send_error(500, 'Explanation could not be generated')
            return


def run(dataset):
    server_address = ('', 8091)
    if dataset == 'ng':
        glove = ExplainableLSTM.load_glove_wordvectors('../wordvectors/glove.6B.50d.txt')
        HTTPRequestHandler.svm = ExplainableSVM.import_model('../trained_models/svm_ng')
        HTTPRequestHandler.lstm = ExplainableLSTM.import_model('../trained_models/lstm_ng',
                                                               glove)
        HTTPRequestHandler.naive = ExplainableNaiveBayes.import_model(
            '../trained_models/naive_ng')
        HTTPRequestHandler.att_lstm = ExplainableAttentionLSTM.import_model(
            '../trained_models/att_lstm_ng', glove)
        server_address = ('', 8091)
    elif dataset == 'tc':
        glove = ExplainableLSTM.load_glove_wordvectors('../wordvectors/tc_custom_trained_vectors.txt')
        HTTPRequestHandler.svm = ExplainableSVM.import_model('../trained_models/svm_tc')
        HTTPRequestHandler.lstm = ExplainableLSTM.import_model('../trained_models/lstm_tc',
                                                               glove)
        HTTPRequestHandler.naive = ExplainableNaiveBayes.import_model(
            '../trained_models/naive_tc')
        HTTPRequestHandler.att_lstm = ExplainableAttentionLSTM.import_model(
            '../trained_models/att_lstm_tc', glove)
        server_address = ('', 8090)
    httpd = HTTPServer(server_address, HTTPRequestHandler)
    # ThreadedHTTPServer does not work yet, because of TensorFlow
    # httpd = ThreadedHTTPServer(server_address, HTTPRequestHandler)
    print('http server is running...')
    httpd.serve_forever()


if __name__ == '__main__':
    if 'tc' in sys.argv:
        run('tc')
    else:
        run('ng')
