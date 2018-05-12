import tensorflow as tf
import pandas as pd
import numpy as np
import string
import cv2
from pcontrol import *


class Predict:
    def __init__(self,
                 sess_id: int,
                 model_path,
                 model_name,
                 prediction_fname='predictions',
                 show_im: bool=True):
        self.model_path = model_path
        self.model_name = model_name
        self.prediction_fname = prediction_fname
        self.id = sess_id
        self.show_im = show_im

        self.raw_predictions = []
        self.predictions = []
        self.Meta = MetaData(sess_id)
        meta = self.Meta.read('data_path', sess_id=sess_id)
        self.Reader = Reader(meta)

        self.pfnames = [self.prediction_fname+'.csv', self.prediction_fname+'_pfile.csv']

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

    def predict(self):
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

        prediction = sess.run(model, feed_dict={x: self.Reader.clean_read()})
        print(prediction)

        pred_labels_int = np.ndarray.tolist(sess.run(tf.argmax(prediction, axis=1)))
        self.raw_predictions = pred_labels_int

        self.int_to_label()
        print(self.predictions)
        self.write_predictions()

    def show(self, p, ph):
        im = cv2.imread(ph)

        cv2.imshow(p, im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def int_to_label(self):
        model_labels = []

        with open(self.model_path + 'labels.txt', 'r') as lbfile:
            for label in lbfile.readlines():
                model_labels.append(label.strip('\n')[0])

        # assigned_labels = self.label_assigner(model_labels)
        # print(assigned_labels)
        assigned_labels = {'A': 0, 'L': 1, 'P': 2}

        for raw_prediction in self.raw_predictions:
            for pred_char, pred_int in assigned_labels.items():
                if raw_prediction == pred_int:
                    self.predictions.append(pred_char)

        return self.predictions

    def write_predictions(self):
        # Re-write data in addition to predicted labels in CSV file; filename is a parameter
        data = self.Reader.clean_read()
        print(len(data))
        print(self.predictions)
        print(self.raw_predictions)
        for pix_data_n in range(len(data)):
            data[pix_data_n].append(self.predictions[pix_data_n])

        with open(self.pfnames[0], 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for im_pix_data in data:
                writer.writerow(im_pix_data)

        # Write image paths together with predicted labels in CSV file
        df = pd.read_csv(self.Meta.read('data_path', sess_id=self.id), header=None)

        raw_rows = df.iterrows()
        rows = []
        for index, row in raw_rows:
            rows.append(list(row.values))

        df = pd.read_csv('metadata/sess/'+str(self.id)+'/impaths.csv', header=None)

        raw_rows = df.iterrows()
        paths = []
        for _, row in raw_rows:
            paths.append(list(row)[0])

        with open(self.pfnames[1], 'w') as pathfile:
            writer = csv.writer(pathfile, delimiter=',')
            for n in range(len(paths)):
                if self.show_im is True:
                    self.show(self.predictions[n], paths[n])
                writer.writerow([paths[n], self.predictions[n]])

    def multilayer_perceptron(self, x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)

        # Hidden layer with sigmoid activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)

        # Hidden layer with sigmoid activation
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.sigmoid(layer_3)

        # Hidden layer with ReLU activation
        layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        layer_4 = tf.nn.relu(layer_4)

        # Output layer with linear activation
        out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
        return out_layer

    def main(self):
        self.predict()
        self.Meta.write(used_model_path__output=self.model_path +
                        self.model_name+'___' +
                        os.getcwd()+'/'+self.pfnames[0] +
                        '__'+os.getcwd()+'/'+self.pfnames[1])


pr = Predict(7, 'training_models/fruit_model4/',
             '-1000',
             show_im=False)
pr.main()

# /home/planetgazer8360/PycharmProjects/TensorFlow/fruit_model4/