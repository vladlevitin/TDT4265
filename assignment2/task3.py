import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, cross_entropy_loss, SoftmaxModel
from task2 import SoftmaxTrainer, calculate_accuracy

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9
    shuffle_data = True
    
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    
    # Task 3
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    use_improved_weight_init = True
    model_improved_weight_init = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_improved_weight_init = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_weight_init, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_weight_init, val_history_improved_weight_init = trainer_improved_weight_init.train(
        num_epochs)
    
    # Adding improved sigmoid
    use_improved_sigmoid = True
    model_improved_sigmoid = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_improved_sigmoid = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_sigmoid, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_sigmoid, val_history_improved_sigmoid = trainer_improved_sigmoid.train(
        num_epochs)

    # Using momentum
    use_momentum = True
    learning_rate = 0.02
    model_use_momentum = SoftmaxModel(
        neurons_per_layer,
        use_momentum,
        use_improved_weight_init)
    trainer_use_momentum = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_use_momentum, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_use_momentum, val_history_use_momentum = trainer_use_momentum.train(
        num_epochs)

    plt.subplot(1, 2, 1)
    utils.plot_loss(val_history["loss"],"Task 2 Model")
    utils.plot_loss(val_history_improved_weight_init["loss"], "Task 3 - Improved weight init")
    utils.plot_loss(val_history_improved_sigmoid["loss"], "Task 3 - Improved weight init+sigmoid")
    utils.plot_loss(val_history_use_momentum["loss"], "Task 3 - Improved weight init+sigmoid+momentum")
    plt.ylim([0, .6])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Loss - Average")

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1.05])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(val_history_improved_weight_init["accuracy"], "Task 3 - Improved weight init")
    utils.plot_loss(val_history_improved_sigmoid["accuracy"], "Task 3 - Improved sigmoid")
    utils.plot_loss(val_history_use_momentum["accuracy"], "Task 3 - Use momentum")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("task3_all_plots.png")
    plt.show()