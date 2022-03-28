import numpy as np
import utils
from task2a import one_hot_encode, pre_process_images, SoftmaxModel, gradient_approximation_test
from task2 import SoftmaxTrainer, calculate_accuracy
import matplotlib.pyplot as plt

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

    # Modify your network here
    neurons_per_layer = [64, 64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)

    # Task 4
    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9

    shuffle_data = True
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    
    # Task 4a

    # Model with 64 neurons in hidden layer for compairison
    neurons_per_layer = [64, 10]

    model_64 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_64 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_64, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_64, val_history_64 = trainer_64.train(num_epochs)

    # Model with 32 neurons in hidden layer
    neurons_per_layer = [32, 10]
    model_32 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_32 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_32, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_32, val_history_32 = trainer_32.train(num_epochs)

    plt.subplot(2, 2, 1)
    plt.ylim([-0.05, 1])
    utils.plot_loss(train_history_64["loss"], "64 hidden layer neurons")
    utils.plot_loss(train_history_32["loss"], "32 hidden layer neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Loss - Average")

    plt.subplot(2, 2, 2)
    plt.ylim([0.95,  1.005])
    utils.plot_loss(train_history_64["accuracy"], "64 hidden layer neurons")
    utils.plot_loss(train_history_32["accuracy"], "32 hidden layer neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Accuracy")

    plt.subplot(2, 2, 3)
    plt.ylim([-0.05, 1])
    utils.plot_loss(val_history_64["loss"], "64 hidden layer neurons")
    utils.plot_loss(val_history_32["loss"], "32 hidden layer neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Loss - Average")

    plt.subplot(2, 2, 4)
    plt.ylim([0.95,  1.005])
    utils.plot_loss(val_history_64["accuracy"], "64 hidden layer neurons")
    utils.plot_loss(val_history_32["accuracy"], "32 hidden layer neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")

    plt.legend()
    plt.tight_layout()
    plt.savefig("task4a.png")
    plt.show()

    # Task 4b

    # Model with 128 neurons in hidden layer
    neurons_per_layer = [128, 10]
    model_128 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_128 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_128, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_128, val_history_128 = trainer_128.train(num_epochs)

    plt.subplot(2, 2, 1)
    plt.ylim([-0.05, 1])
    utils.plot_loss(train_history_64["loss"], "64 hidden layer neurons")
    utils.plot_loss(train_history_128["loss"], "128 hidden layer neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Loss - Average")

    plt.subplot(2, 2, 2)
    plt.ylim([0.95,  1.005])
    utils.plot_loss(train_history_64["accuracy"], "64 hidden layer neurons")
    utils.plot_loss(train_history_128["accuracy"], "128 hidden layer neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Accuracy")

    plt.subplot(2, 2, 3)
    plt.ylim([-0.05, 1])
    utils.plot_loss(val_history_64["loss"], "64 hidden layer neurons")
    utils.plot_loss(val_history_128["loss"], "128 hidden layer neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Loss - Average")

    plt.subplot(2, 2, 4)
    plt.ylim([0.95,  1.005])
    utils.plot_loss(val_history_64["accuracy"], "64 hidden layer neurons")
    utils.plot_loss(val_history_128["accuracy"], "128 hidden layer neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("task4b.png")
    plt.show()
    
    
    # Task 4d
    
    # Neural network of two layers with 60 neuron in each
    neurons_per_layer = [60, 60, 10]
    model = SoftmaxModel(
        neurons_per_layer,
        use_momentum,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    plt.subplot(2, 2, 1)
    plt.ylim([-0.05, 1.5])
    utils.plot_loss(train_history_64["loss"], "[64, 10]")
    utils.plot_loss(train_history["loss"], "[60, 60, 10]")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Loss - Average")

    plt.subplot(2, 2, 2)
    plt.ylim([0.95,  1.005])
    utils.plot_loss(train_history_64["accuracy"], "[64, 10]")
    utils.plot_loss(train_history["accuracy"], "[60, 60, 10]")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Accuracy")

    plt.subplot(2, 2, 3)
    plt.ylim([-0.05, 1.5])
    utils.plot_loss(val_history_64["loss"], "[64, 10]")
    utils.plot_loss(val_history["loss"], "[60, 60, 10]")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Loss - Average")

    plt.subplot(2, 2, 4)
    plt.ylim([0.95,  1.005])
    utils.plot_loss(val_history_64["accuracy"], "[64, 10]")
    utils.plot_loss(val_history["accuracy"], "[60, 60, 10]")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("task4d.png")
    plt.show()

    # Task 4e

    # Neural network of ten layers with 64 neuron in each
    neurons_per_layer = [64] * 10
    neurons_per_layer.append(10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_momentum,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    plt.subplot(2, 2, 1)
    plt.ylim([-0.05, 1.5])
    utils.plot_loss(train_history_64["loss"], "[64, 10]")
    utils.plot_loss(train_history["loss"], "[64] * 10")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Loss - Average")

    plt.subplot(2, 2, 2)
    plt.ylim([0.95,  1.005])
    utils.plot_loss(train_history_64["accuracy"], "[64, 10]")
    utils.plot_loss(train_history["accuracy"], "[64] * 10")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Accuracy")

    plt.subplot(2, 2, 3)
    plt.ylim([-0.05, 1.5])
    utils.plot_loss(val_history_64["loss"], "[64, 10]")
    utils.plot_loss(val_history["loss"], "[64] * 10")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Loss - Average")

    plt.subplot(2, 2, 4)
    plt.ylim([0.95,  1.005])
    utils.plot_loss(val_history_64["accuracy"], "[64, 10]")
    utils.plot_loss(val_history["accuracy"], "[64] * 10")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("task4e.png")
    plt.show()
    

