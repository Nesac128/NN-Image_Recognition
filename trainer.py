import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import time


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Train:
    def __init__(self,
                 training_data_path,
                 n_columns,
                 n_label_types,
                 model_store_path,
                 optimizer='GradientDescent',
                 n_perceptrons_layer: int=100,
                 epochs: int=150,
                 learning_rate: float=0.2,
                 train_test_split: float=0.2):

        # Storing parameters
        self.training_data_path = training_data_path
        self.n_columns = n_columns
        self.n_classes = n_label_types
        self.model_store_path = model_store_path
        self.optimizer = optimizer
        self.n_perceptrons_layer = n_perceptrons_layer
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_test_split = train_test_split

        # Defining training variables
        self.X, self.Y = self.read_dataset()

        self.x_shape = self.X.shape
        self.y_shape = self.Y.shape

        self.weights = None
        self.biases = None

    def read_dataset(self):
        df = pd.read_csv(self.training_data_path, header=None)
        X = df[df.columns[0:self.n_columns]].values
        y = df[df.columns[self.n_columns]]

        # Encode the dependent variable
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)
        print(type(y))
        Y = self.one_hot_encoder(y)
        print("Tensor shape for data: ", X.shape)
        print("Tensor shape for labels: ", Y.shape)
        return X, Y

    def one_hot_encoder(self, labels):
        n_labels = len(labels)
        n_unique_labels = len(np.unique(labels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        one_hot_encode[np.arange(n_labels), labels] = 1
        return one_hot_encode

    def display(self, dptype):
        if dptype == 'pre-training':
            print(Colors.BOLD + Colors.BLUE + """
            =============================================================
                          INITIALIZING VARIABLES & TRAINING
            
            INFORMATION
            learning_rate: """, self.learning_rate, """
            epochs: """, self.epochs, """        
            perceptrons_per_layer: """, self.n_perceptrons_layer, """
            =============================================================            
            """)

        elif dptype == 'post-training':
            print(Colors.BOLD + Colors.GREEN + """
            =================================================================
                              TRAINING STAGE HAS FINALIZED
            
            Calculating final Mean Squared Error (MSE), Accuracy and Cost...
            =================================================================
            """)

    def multilayer_neural_network(self, x, weights, biases):
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
        layer_4 = tf.nn.sigmoid(layer_4)

        layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
        layer_5 = tf.nn.relu(layer_5)

        # Output layer with linear activation
        out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
        return out_layer

    def train(self):
        # Randomly mix the order of rows
        X, Y = shuffle(self.X, self.Y, random_state=1)
        print("Splitting dataset into training and test sections...")
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=self.train_test_split,
                                                            random_state=415)

        print("Train_data shape: ", train_x.shape, "  Train_labels shape: ", train_y.shape,
              "  Test_data shape: ", test_x.shape)
        time.sleep(1.5)

        n_dim = self.X.shape[1]
        print("Data dimensions: ", n_dim)
        time.sleep(1)

        perceptron_n_layer1 = self.n_perceptrons_layer
        perceptron_n_layer2 = self.n_perceptrons_layer
        perceptron_n_layer3 = self.n_perceptrons_layer
        perceptron_n_layer4 = self.n_perceptrons_layer
        perceptron_n_layer5 = self.n_perceptrons_layer

        x = tf.placeholder(tf.float32, [None, n_dim], name='x')
        y = tf.placeholder(tf.float32, [None, self.n_classes], name='y')

        weights = {
            'h1': tf.Variable(tf.truncated_normal([n_dim, perceptron_n_layer1]), name='weights1'),
            'h2': tf.Variable(tf.truncated_normal([perceptron_n_layer1, perceptron_n_layer2]), name='weights2'),
            'h3': tf.Variable(tf.truncated_normal([perceptron_n_layer2, perceptron_n_layer3]), name='weights3'),
            'h4': tf.Variable(tf.truncated_normal([perceptron_n_layer3, perceptron_n_layer4]), name='weights4'),
            'h5': tf.Variable(tf.truncated_normal([perceptron_n_layer4, perceptron_n_layer5]), name='weights5'),
            'out': tf.Variable(tf.truncated_normal([perceptron_n_layer5, self.n_classes]), name='weights6')
        }

        biases = {
            'b1': tf.Variable(tf.truncated_normal([perceptron_n_layer1]), name='biases'),
            'b2': tf.Variable(tf.truncated_normal([perceptron_n_layer2]), name='biases'),
            'b3': tf.Variable(tf.truncated_normal([perceptron_n_layer3]), name='biases'),
            'b4': tf.Variable(tf.truncated_normal([perceptron_n_layer4]), name='biases'),
            'b5': tf.Variable(tf.truncated_normal([perceptron_n_layer5]), name='biases'),
            'out': tf.Variable(tf.truncated_normal([self.n_classes]), name='biases')
        }

        model = self.multilayer_neural_network(x, weights, biases)

        init = tf.global_variables_initializer()

        cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
        trainer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost_function)

        sess = tf.Session()
        sess.run(init)

        mse_history = []
        accuracy_history = []
        cost_history = np.empty(shape=[1], dtype=float)

        saver = tf.train.Saver()

        self.display('pre-training')
        for epoch in range(self.epochs):
            sess.run(trainer, feed_dict={x: train_x, y: train_y})

            # Calculate cost/loss and append result to cost_history
            cost = sess.run(cost_function, feed_dict={x: train_x, y: train_y})
            cost_history = np.append(cost_history, cost)
            correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Make prediction using test_data, calculate Mean Squared Error (MSE) and append it to mse_history
            pred_y = sess.run(model, feed_dict={x: test_x})
            mse = tf.reduce_mean(tf.square(pred_y - test_y))
            mse_ = sess.run(mse)
            print("Test_data Mean Squared Error (MSE): ", mse_)
            mse_history.append(mse_)

            # Calculate Training Accuracy and append to accuracy_history
            accuracy = (sess.run(accuracy, feed_dict={x: train_x, y: train_y}))
            accuracy_history.append(accuracy)

            print('EPOCH ', epoch, '  --  ', 'Cost: ', cost, "  --  MSE: ", mse_, "  --  Training Accuracy: ", accuracy)

        self.display('post-training')

        correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        pred_test_y = sess.run(model, feed_dict={x: test_x})
        print("Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y: test_y})))
        mse = tf.reduce_mean(tf.square(pred_test_y - test_y))
        mse_ = sess.run(mse)
        print("Final MSE (Mean Squared Error): ", mse_)
        print("Final cost: ", (sess.run(cost_function, {x: test_x, y: test_y})))

        saver.save(sess, self.model_store_path, global_step=1000)
