import tensorflow as tf
import pandas as pd
import numpy as np
import string
import csv
import cv2


class Predict:
    def __init__(self,
                 model_path,
                 raw_data_for_prediction_path,
                 model_name,
                 prediction_fname='predictions.csv',
                 show_im: bool=True):
        self.model_path = model_path
        self.data_to_classify_path = raw_data_for_prediction_path
        self.model_name = model_name
        self.prediction_fname = prediction_fname

        self.raw_predictions = None
        self.predictions = []

        self.sess_id = 0

    def get_data(self):
        df = pd.read_csv(self.data_to_classify_path, header=None)
        X = df[df.columns[0:len(df.columns)]].values
        return X

    def label_assigner(self, labels):
        unique_labels = []
        for n_label in range(len(labels)):
            if n_label == 0:
                unique_labels.append(labels[n_label])
            else:
                if labels[n_label] == labels[n_label - 1]:
                    pass
                else:
                    unique_labels.append(labels[n_label])

        labels_UP = {}
        labels_DOWN = {}

        int_to_label = {}

        ln = 0
        for letter in list(string.ascii_uppercase):
            labels_UP[letter] = ln
            ln += 1

        ln = 0

        for letter in list(string.ascii_lowercase):
            labels_DOWN[letter] = ln
            ln += 1

        label_counter = 0

        for n_label in range(len(sorted(unique_labels))):
            int_to_label[sorted(unique_labels)[n_label]] = label_counter
            label_counter += 1

        return int_to_label

    def main(self):
        sess = tf.Session()

        # Create saver
        saver = tf.train.import_meta_graph(self.model_path + self.model_name + '.meta')

        # Attempt to restore model for prediction
        saver.restore(sess, tf.train.latest_checkpoint(self.model_path + './'))
        print("Trained model has been restored successfully!")

        x = tf.placeholder(tf.float32, [None, sess.run('n_dim:0')])

        w1, w2, w3, w4, w5 = sess.run(('weights1:0', 'weights2:0', 'weights3:0', 'weights4:0', 'weights5:0'))
        b1, b2, b3, b4, b5 = sess.run(('biases1:0', 'biases2:0', 'biases3:0', 'biases4:0', 'biases5:0'))

        weights = {
            'h1': tf.convert_to_tensor(w1),
            'h2': tf.convert_to_tensor(w2),
            'h3': tf.convert_to_tensor(w3),
            'h4': tf.convert_to_tensor(w4),
            'out': tf.convert_to_tensor(w5)
        }

        biases = {
            'b1': tf.convert_to_tensor(b1),
            'b2': tf.convert_to_tensor(b2),
            'b3': tf.convert_to_tensor(b3),
            'b4': tf.convert_to_tensor(b4),
            'out': tf.convert_to_tensor(b5)
        }

        model = self.multilayer_perceptron(x, weights, biases)

        prediction = sess.run(model, feed_dict={x: self.get_data()})

        pred_labels_int = np.ndarray.tolist(sess.run(tf.argmax(prediction, axis=1)))
        self.raw_predictions = pred_labels_int

        self.int_to_label()
        self.write_predictions()

    def int_to_label(self):
        model_labels = []

        with open(self.model_path + 'labels.txt', 'r') as lbfile:
            for label in lbfile.readlines():
                model_labels.append(label.strip('\n')[0])

        assigned_labels = self.label_assigner(model_labels)

        for raw_prediction in self.raw_predictions:
            for pred_char, pred_int in assigned_labels.items():
                if raw_prediction == pred_int:
                    self.predictions.append(pred_char)

        return self.predictions

    def write_predictions(self):
        # Re-write data in addition to predicted labels in CSV file; filename is a parameter
        raw = self.get_data()
        p_raw = list(np.ndarray.tolist(raw))

        for pix_dat_n in range(len(raw)):
            p_raw[pix_dat_n].append(self.predictions[pix_dat_n])

        tfl = p_raw
        print(p_raw)
        for pr in tfl:
            print(pr[0])

        with open(self.prediction_fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for pr in tfl:
                writer.writerow(pr)

        # Write image paths together with predicted labels in CSV file
        df = pd.read_csv(self.data_to_classify_path, header=None)

        raw_rows = df.iterrows()
        rows = []
        for index, row in raw_rows:
            rows.append(list(row.values))

        for row in rows:
            try:
                self.sess_id = int(row[0])
            except ValueError:
                continue

        df = pd.read_csv('cpaths.csv', header=None)

        raw_rows = df.iterrows()
        paths = []

        for index, path in raw_rows:
            paths.append(list(path.values))

        with open(self.prediction_fname+'_cpaths.csv', 'w') as pathfile:
            writer = csv.writer(pathfile, delimiter=',')
            print(paths)
            for n in range(len(paths)):
                print(self.predictions)
                writer.writerow([paths[n], self.predictions[n]])

    def multilayer_perceptron(self, x, weights, biases):
        # Hidden layer with RELU activationsd
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)

        # Hidden layer with sigmoid activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)

        # Hidden layer with sigmoid activation
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.sigmoid(layer_3)

        # Hidden layer with RELU activation
        layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        layer_4 = tf.nn.relu(layer_4)

        # Output layer with linear activation
        out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
        return out_layer


p = Predict('/home/planetgazer8360/PycharmProjects/TensorFlow/my_test_model4/',
            '/home/planetgazer8360/PycharmProjects/TensorFlow/test_apple_predicter.csv', '-1000')
p.main()
