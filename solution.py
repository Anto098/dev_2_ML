import numpy as np
#! REMOVE THESE BEFORE SUBMITTING !#
from icecream import ic   # library to pretty print, call: ic(variable) is equivalent to print(f"variable:{variable}")
# import pandas as pd       # wrapper for np.arrays, used in a similar way to an SQL table (join,  sort, aggregate, etc)
#! ############################## !#

class SVM:
    def __init__(self, eta, C, niter, batch_size, verbose):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        y : numpy array of shape (n,)
        m : int (num_classes)
        returns : numpy array of shape (n, m)
        """
        # oh_labels = one hot labels (but 0s are replaced by -1s)
        oh_labels = np.zeros((y.shape[0], m), dtype=int)
        oh_labels.fill(-1)
        for (i, y_i) in enumerate(y):
            oh_labels[i, y_i] = 1
        
        return oh_labels

    def compute_loss(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        w : numpy array of shape (num_features, num_classes) (1 col per class)
        
        
        if they were not matrices:
        
        x : np array of shape (, num_features)
        y : np array of shape (, num_classes)
        w : np array of shape (num_features, )
        
        xw : (, num_features) x (num_features, ) = (1,1)
        xwy : (1,1) x (, num_classes) = (1, num_classes)
        returns : float
        """
        # ic(self.w.shape)
        # y is already encoded in 1 hot so we don't need to modify it
        xw = np.dot(x, self.w)  # shape: (minibatch size, num_classes)
        xwy = np.dot(xw.T, y)   # (minibatch size, num_classes).T x (minibatch size, num_classes) = (num_classes, num_classes)
        p = 2 - xwy 
        max = np.maximum(np.zeros_like(p), p)
        loss = np.power(max, 2)
        mean_loss_per_class = np.mean(loss, axis=1)
        sum_loss = np.sum(mean_loss_per_class)
        return sum_loss

    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : numpy array of shape (num_features, num_classes)
        """
        
        pass

    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (num_examples_to_infer, num_features)
        returns : numpy array of shape (num_examples_to_infer, num_classes)
        """
        pass

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (num_examples, num_classes)
        y : numpy array of shape (num_examples, num_classes)
        returns : float
        """
        pass

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, num_features)
        y_train : numpy array of shape (number of training examples, num_classes)
        x_test : numpy array of shape (number of training examples, nujm_features)
        y_test : numpy array of shape (number of training examples, num_classes)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x, y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train, y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test, y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print(f"Iteration {iteration} | Train loss {train_loss:.04f} | Train acc {train_accuracy:.04f} |"
                      f" Test loss {test_loss:.04f} | Test acc {test_accuracy:.04f}")

            # Record losses, accs
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            test_losses.append(test_loss)
            test_accs.append(test_accuracy)

        return train_losses, train_accs, test_losses, test_accs


# DO NOT MODIFY THIS FUNCTION
def load_data():
    # Load the data files
    print("Loading data...")
    x_train = np.load("x_train_cifar10_reduced.npz")["x_train"]
    x_test = np.load("x_test_cifar10_reduced.npz")["x_test"]
    y_train = np.load("y_train_cifar10_reduced.npz")["y_train"]
    y_test = np.load("y_test_cifar10_reduced.npz")["y_test"]

    # normalize the data
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # add implicit bias in the feature
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

    return x_train, y_train, x_test, y_test

def test_3a():
    svm = SVM(eta=0.0001, C=2, niter=200, batch_size=5000, verbose=False)
    oh_labels = svm.make_one_versus_all_labels(y=np.array([1,0,2]), m=4)
    # ic(oh_labels)

def test_3b(x_train, y_train):
    svm = SVM(eta= 0.0001, C= 2, niter= 1, batch_size= 5000, verbose= False)
    
    x_train = x_train[:5000, :]
    y_train = y_train[:5000]
    y_train = np.random.randint(0, 8, y_train.shape) # put 8 classes to mimic gradescope
    
    num_features = x_train.shape[1]
    m = y_train.max() + 1   # num_classes
    svm.m = m
    y_train = svm.make_one_versus_all_labels(y_train, m)
    
    svm.w = np.zeros([num_features, m])
    loss = svm.compute_loss(x_train, y_train)
    ic(loss)



if __name__ == "__main__":

    # x_train: (20 000, 3 073)
    # y_train: (20 000,)
    # x_test: (4 000, 3 073)
    # y_test: (4 000,)
    # last col (3072) is bias (1 for each row/example)
    x_train, y_train, x_test, y_test = load_data()

    print("Fitting the model...")
    # svm = SVM(eta=0.0001, C=2, niter=200, batch_size=5000, verbose=False)
    # test_3a()
    test_3b(x_train, y_train)
    
    # train_losses, train_accs, test_losses, test_accs = svm.fit(x_train, y_train, x_test, y_test)

    # # to infer after training, do the following:
    # y_inferred = svm.infer(x_test)

    ## to compute the gradient or loss before training, do the following:
    # y_train_ova = svm.make_one_versus_all_labels(y_train, 8) # one-versus-all labels
    # svm.w = np.zeros([3073, 8])
    # grad = svm.compute_gradient(x_train, y_train_ova)
    # loss = svm.compute_loss(x_train, y_train_ova)
    
    

