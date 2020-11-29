import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def plot_image(
    i: int, predictions_array: np.ndarray, true_label: np.ndarray, img: np.ndarray
) -> None:
    """Plots an image with its predicted and true label

    Args:
        i (int): image index in the data set
        predictions_array (np.ndarray): our model predictions, array of probabilities of classes in CLASS_NAME
        true_label (np.ndarray): array of correct labels
        img (np.ndarray): data set of images
    """
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            CLASS_NAMES[predicted_label],
            100 * np.max(predictions_array),
            CLASS_NAMES[true_label],
        ),
        color=color,
    )


def plot_value_array(
    i: int, predictions_array: np.ndarray, true_label: np.ndarray
) -> None:
    """Plots the probability of each class according to our model predictions

    Args:
        i (int): index of the example
        predictions_array (np.ndarray): model predictions
        true_label (np.ndarray): array of labels
    """
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


def visualise(images: np.ndarray, labels: np.ndarray, n: int = 25) -> None:
    """Visualises the first n entries in a set.

    Args:
        images (np.ndarray): an array of 28x28px images.
        labels (np.ndarray): an array of numeric labels for the images
        n (int): the number of entries to visualise, must be <=25.

    Returns:
        None
    """
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(CLASS_NAMES[labels[i]])
    plt.show()

    return None


def preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and processes data from fashion_mnist dataset

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: train_images, test_images are arrays of 60,000/10,000 28x28px images
                    train_labels, test_labels are arrays of numbers 0-9 representing classes in CLASS_NAMES
    """
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # scaling images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels


def verify_predictions(
    predictions: np.ndarray,
    test_labels: np.ndarray,
    test_images: np.ndarray,
    num_rows: int = 5,
    num_cols: int = 3,
) -> None:
    """Checks predictions for the first num_rows * num_cols images.

    Args:
        predictions (np.ndarray): array of model predictions
        test_labels (np.ndarray): labels for test set
        test_images (np.ndarray): images in test set
        num_rows (int, optional): number of plots per column. Defaults to 5.
        num_cols (int, optional): number of plots per row. Defaults to 3.
    """
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = preprocess_data()

    # visualise(train_images, train_labels, 25)
    # visualise(test_images, test_labels, 25)

    # define model architecture
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10),
        ]
    )

    # compile model
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # train model
    model.fit(train_images, train_labels, epochs=10)

    # measure metrics
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print("\nTest accuracy: {}".format(test_acc))

    # wrap around a Softmax so we get actual probabilities
    probability_model = keras.Sequential([model, keras.layers.Softmax()])

    # predict and verify predictions for test set
    predictions = probability_model.predict(test_images)
    verify_predictions(predictions, test_labels, test_images)
