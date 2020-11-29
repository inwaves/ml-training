import matplotlib.pyplot as plt
import re
import string
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, losses, preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

TRAIN_DIR = "aclImdb/train"
TEST_DIR = "aclImdb/test"
AUTOTUNE = tf.data.experimental.AUTOTUNE
EMBEDDING_DIM = 16
MAX_FEATURES = 10000
SEQUENCE_LENGTH = 250
EPOCHS = 10


class SentimentAnalyser:
    """
    FIXME: About the dataset

    """

    def vectorise_text(self, text, label):
        """Applies the text vectorisation layer.

        Args:
            text (string): the text to vectorise
            label (int): the label (1 pos, 0 neg)

        Returns:
            [type]: [description]
        """
        # this adds the text to a list, since layers work on lists
        text = tf.expand_dims(text, -1)
        return self.vectorise_layer(text), label

    def custom_standardisation(self, input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
        return tf.strings.regex_replace(
            stripped_html, "[%s]" % re.escape(string.punctuation), ""
        )

    def __init__(
        self,
        batch_size,
        seed,
        vectorisation_max_features=MAX_FEATURES,
        vectorisation_sequence_length=SEQUENCE_LENGTH,
    ):
        """Constructor for one SentimentAnalyser model.

        Args:
            batch_size (int): size of the batch when loading the dataset,
            seed (int): the seed of the dataset, for reproducibility
        """
        # load training set
        raw_train_ds = keras.preprocessing.text_dataset_from_directory(
            TRAIN_DIR,
            batch_size=batch_size,
            validation_split=0.2,
            subset="training",
            seed=seed,
        )

        # load validation set
        raw_val_ds = keras.preprocessing.text_dataset_from_directory(
            TRAIN_DIR,
            batch_size=batch_size,
            validation_split=0.2,
            subset="validation",
            seed=seed,
        )

        # load test set
        raw_test_ds = keras.preprocessing.text_dataset_from_directory(
            TEST_DIR, batch_size=batch_size
        )

        # layer to standardise, tokenise, vectorise the data
        # creates unique integers for all tokens
        self.vectorise_layer = TextVectorization(
            max_tokens=vectorisation_max_features,
            standardize=self.custom_standardisation,
            output_mode="int",
            output_sequence_length=vectorisation_sequence_length,
        )

        # fit the preprocessing layer to the training data, without labels
        train_text = raw_test_ds.map(lambda x, y: x)
        self.vectorise_layer.adapt(train_text)

        self.train_ds = raw_train_ds.map(self.vectorise_text)
        self.test_ds = raw_test_ds.map(self.vectorise_text)
        self.val_ds = raw_val_ds.map(self.vectorise_text)

    def print_data(self, raw_train_ds, n=1):
        """Prints the first 3 reviews of the first n films.txt

        Args:
            raw_train_ds (tf.dataset): the dataset,
            n (int): the amount of entries to get the reviews for
        """
        for text_batch, label_batch in raw_train_ds.take(n):
            for i in range(3):
                print("Review: {}".format(text_batch.numpy()[i]))
                print("Label: {}".format(label_batch.numpy()[i]))

    def create_model(self):
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.test_ds = self.test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        self.model = keras.Sequential(
            [
                layers.Embedding(MAX_FEATURES + 1, EMBEDDING_DIM),
                layers.Dropout(0.2),
                layers.GlobalAveragePooling1D(),
                layers.Dropout(0.2),
                layers.Dense(1),
            ]
        )

        self.model.summary()

        self.model.compile(
            loss=losses.BinaryCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=tf.metrics.BinaryAccuracy(threshold=0.0),
        )

        # history = self.model.fit(
        #     self.train_ds, validation_data=self.val_ds, epochs=EPOCHS
        # )


if __name__ == "__main__":
    sa_model = SentimentAnalyser(42, 42, 10000, 250)
    sa_model.create_model()

    loss, accuracy = sa_model.model.evaluate(sa_model.test_ds)
    print("Loss: {}\nAccuracy: {}".format(loss, accuracy))
