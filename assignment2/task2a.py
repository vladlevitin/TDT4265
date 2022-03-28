from random import sample
from turtle import st
import numpy as np
from sympy import GramSchmidt
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    # Mean and std. Save so they are the same for training, validation and test
    mean = 33.55274553571429
    std = 78.87550070784701

    # Normalize
    X = (X-mean)/std

    # Bias trick
    X = np.append(X, np.zeros([X.shape[0], 1]), axis= 1)
    
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    loss = -np.average(np.sum(targets*np.log(outputs), axis=1))

    return loss

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    denominator = np.exp(x).sum(axis=1)[np.newaxis, :].T  # [batch_size, 1]
    numerator = np.exp(x)  # [batch size, num_outputs]
    # Elementwise numerator[batch,:]/denominator[batch]
    res = numerator/denominator

    assert np.sum(res, axis=1).all() == 1,\
        f"Sum of softmax is not 1"

    return res

def improved_sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.7159*np.tanh((2/3)*z)

def dimproved_sigmoid(a: np.ndarray) -> np.ndarray:
    return 1.7159*(2.0/3.0)*(1 - ((1/1.7159)*a)**2)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0/(1.0+np.exp(-z))

def dsigmoid(a: np.ndarray) -> np.ndarray:
    return a*(1.0-a)

class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            # Task 3a
            if use_improved_weight_init:
                w = np.random.normal(0, 1/np.sqrt(size), w_shape)
            else:
                # Task 2c
                # Initialize the weights between -1 and 1
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]
        self.hidden_layer_output = [
            None for i in range(len(neurons_per_layer)-1)]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """

        # Calculate output for input layer
        z = X @ self.ws[0]
        if self.use_improved_sigmoid:
            a = improved_sigmoid(z)
        else:
            a = sigmoid(z)

        self.hidden_layer_output[0] = a

        # Go through all the hidden layers and calculate output
        for layer_idx in range(1, len(self.hidden_layer_output)):
            # Here j and k denotes thow sequential layers
            x_j = self.hidden_layer_output[layer_idx - 1]
            w_jk = self.ws[layer_idx]

            z_k = x_j @ w_jk

            if self.use_improved_sigmoid:
                a_k = improved_sigmoid(z_k)
            else:
                a_k = sigmoid(z_k)
            
            self.hidden_layer_output[layer_idx] = a_k
        
        # Calculate output for last layer
        z = self.hidden_layer_output[-1] @ self.ws[-1]
        output = softmax(z)

        return output

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # Task 2b
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        batch_size = X.shape[0]

        # Compute delta for output layer
        delta = -(targets - outputs)

        # Gradient of output
        output_grad = delta.T @ self.hidden_layer_output[-1] / batch_size

        # Store this gradient last in self.grads
        self.grads[-1] = output_grad.T

        # Find gradients for hidden layers. Start with the one closest to output
        # and end with the one closest to input
        for layer_idx in range(len(self.hidden_layer_output) - 1, 0, -1):
            # Update delta as given in task 1a
            # Here j and k denotes two consequtive layers

            # Input to layer k
            x_k = self.hidden_layer_output[layer_idx - 1]
            
            # Output layer j
            z_j = self.hidden_layer_output[layer_idx]

            # Weights from layer j to layer 
            w_kj = self.ws[layer_idx + 1]

            if self.use_improved_sigmoid:
                delta = dimproved_sigmoid(z_j) * (delta @ w_kj.T)
            else:
                delta = dsigmoid(z_j) * (delta @ w_kj.T)

            self.grads[layer_idx] =  x_k.T @ delta / batch_size

        # Compute grad for input layer using X
        # Weight from input layer to first hidden layer
        w_ji = self.ws[1]

        # Output from input layer
        z_j = self.hidden_layer_output[0]

        if self.use_improved_sigmoid:
            delta = dimproved_sigmoid(z_j) * (delta @ w_ji.T)
        else:
            delta = dsigmoid(z_j) * (delta @ w_ji.T)
        
        self.grads[0] = X.T @ delta / batch_size

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]

    def print_num_parameters(self) -> None:
        layer = 0
        total_params = 0
        for w in self.ws:
            total_params += w.shape[0]*w.shape[1]
            print("w[" + str(layer) + "]: " + str(w.shape))
            layer += 1
        print("Total parameters: " + str(total_params))


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    one_hot_encoded = np.zeros((Y.shape[0], num_classes))
    for i, sample in enumerate(Y):
        one_hot_encoded[i][sample[0]] = 1

    return one_hot_encoded


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
