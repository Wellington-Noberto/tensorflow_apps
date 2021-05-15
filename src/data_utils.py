import os
import sys
import tensorflow as tf
from keras.models import load_model

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


if __name__ == '__main__':
    from data_processing import data_preprocess_test
else:
    PACKAGE_PARENT = '...'
    SCRIPT_DIR = os.path.join(os.getcwd(), PACKAGE_PARENT)
    sys.path.append(SCRIPT_DIR)
    from src.data_processing import data_preprocess_test


def decode_img(img_path, img_width, img_height, num_channels):
    """ Reads and converts images

    Args:
        img_path (tf.image): Input image
        img_width (int): Image width
        img_height (int): Image height
        num_channels (int): Number of channels on the images
    Returns:
        img (tf.image): Image after preprocessing
    """
    # Reads the image
    img = tf.io.read_file(img_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, num_channels)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.image.resize(img, (img_height, img_width))
    # Returns a tensor with a length 1 axis inserted at index axis
    img = tf.expand_dims(img, 0)

    return img


def model_predict(model_path, img_path, img_width, img_height, num_channels, class_names=None):
    """ Classifies a single image

    Args:
        model_path (tf.h5): CNN model to be trained
        img_path (str): Path to the image. Preferably .jpg or .png
        img_width (int): Image width
        img_height (int): Image height
        num_channels (int): Number of channels on the images
        class_names (list of str):

    Returns:

    """
    # load model
    model = load_model(model_path)
    # preprocess path of image
    img = decode_img(img_path, img_width, img_height, num_channels)
    # predict one image
    prediction = model.predict(img)

    if class_names is None:
        # returns the name of the predicted class
        predicted_class = class_names[np.argmax(prediction, 1)[0]]
        return predicted_class
    else:
        return prediction


def confusion_matrix_gerenate(y_true, y_pred, class_names, normalized=False):
    """ Generates a confusion matrix

    Args:
        y_true: list of true labels
        y_pred: list of predicted labels
        class_names: list of classe names
        normalized (bool):

    Returns:

    """

    conf_matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    if normalized:
        conf_matrix = conf_matrix / conf_matrix.astype(np.float).sum(axis=1)

    sns.heatmap(conf_matrix, cbar=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot=True)
    plt.ylabel('Classe verdadeira')
    plt.xlabel('Classe prevista')
    plt.savefig('tests/confusion_matrix.png')

    return conf_matrix


def model_evaluate(model_path, input_path, img_height, img_width, num_channels, class_names):
    """

    Args:
        model_path:
        input_path:
        img_height:
        img_width:
        num_channels:
        class_names:

    Returns:

    """

    # load model
    model = load_model(model_path)
    test_data = data_preprocess_test(input_path, img_height, img_width, num_channels, class_names)
    # Accuracy and loss
    test_scores = model.evaluate(test_data)

    y_true = tf.argmax(tf.concat([y for x, y in test_data], axis=0), axis= 1)
    y_pred = tf.argmax(model.predict(test_data), axis=1)

    # Using labels instead of index
    class_names = np.array(class_names)
    y_true = class_names[y_true.numpy()]
    y_pred = class_names[y_pred.numpy()]

    conf_matrix = confusion_matrix_gerenate(y_true, y_pred, class_names)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(class_report).transpose().to_csv('classification_report.csv')
    class_report = classification_report(y_true, y_pred, output_dict=False)

    return test_scores, conf_matrix, class_report
