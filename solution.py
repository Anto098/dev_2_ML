import numpy as np


class SVM:
    def __init__(self, eta, C, niter, batch_size, verbose, hinge_offset=2):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose
        self.hinge_offset = hinge_offset

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
        x : numpy array of shape (minibatch_size, num_features)
        y : numpy array of shape (minibatch_size, num_classes)
        returns : float
        """
        # y is already encoded in 1 hot so we don't need to modify it
        xw = np.dot(x, self.w)  # shape: (minibatch size, num_classes)
        xwy = np.multiply(xw, y)
        p = self.hinge_offset - xwy

        max_func = np.maximum(np.zeros_like(p), p)
        squared_l = np.power(max_func, 2)
        mean_loss_per_class = np.sum(squared_l, axis=1)

        regularization = self.C * (self.w ** 2).sum() / 2
        return np.mean(mean_loss_per_class) + regularization

    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : numpy array of shape (num_features, num_classes)
        """
        xw = np.dot(x, self.w)  # shape: (minibatch size, num_classes)

        hinge_grad = 2 * x.T @ (xw - self.hinge_offset * y) / x.shape[0]
        regul_grad = self.C * self.w
        return hinge_grad + regul_grad

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
        assert self.w.shape[0] == x.shape[1]
        scores = x @ self.w
        pred_classes = scores.argmax(axis=1)
        pred_ova = -np.ones(scores.shape)
        pred_ova[np.arange(scores.shape[0]), pred_classes] = 1
        return pred_ova

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (num_examples, num_classes)
        y : numpy array of shape (num_examples, num_classes)
        returns : float
        """
        assert y_inferred.shape == y.shape
        return (y_inferred[np.where(y == 1)] == 1).sum() / y.shape[0]

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

        # self.w : numpy array of shape (num_features, num_classes)
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

            # Record losses, accuracies
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


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_data()

    print("Fitting the model...")
    # svm = SVM(eta=0.0001, C=2, niter=200, batch_size=5000, verbose=False)
    # train_losses, train_accs, test_losses, test_accs = svm.fit(x_train, y_train, x_test, y_test)

    # # to infer after training, do the following:
    # y_inferred = svm.infer(x_test)

    ## to compute the gradient or loss before training, do the following:
    # y_train_ova = svm.make_one_versus_all_labels(y_train, 8) # one-versus-all labels
    # svm.w = np.zeros([3073, 8])
    # grad = svm.compute_gradient(x_train, y_train_ova)
    # loss = svm.compute_loss(x_train, y_train_ova)
