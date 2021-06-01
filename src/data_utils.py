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


def read_test_img(img_path, img_height, img_width, num_channels):
    """ Reads and converts images

    Args:
        img_path (tf.image): Input image
        img_height (int): Image height
        img_width (int): Image width
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


def model_predict(model_path, img_path, img_height, img_width, num_channels, class_names=None):
    """ Classifies a single image

    Args:
        model_path (tf.h5): CNN model to be trained
        img_path (str): Path to the image. Preferably .jpg or .png
        img_height (int): Image height
        img_width (int): Image width
        num_channels (int): Number of channels on the images
        class_names (list of str): List of classes

    Returns:

    """
    # load model
    model = load_model(model_path)
    # preprocess path of image
    img = read_test_img(img_path, img_width, img_height, num_channels)
    # predict one image
    prediction = model.predict(img)

    if class_names is None:
        # returns the name of the predicted class
        predicted_class = class_names[np.argmax(prediction, 1)[0]]
        return predicted_class

    elif class_names == 'segmentation':
        img = tf.reshape(img, (128, 128, 3))
        mask = np.argmax(prediction, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        return img, mask[0]

    else:
        return prediction


def plot_segmentation_mask(model_path, img_path, img_height, img_width, num_channels):

    img, mask = model_predict(model_path, img_path, img_height, img_width, num_channels, class_names='segmentation')

    plt.figure(figsize=(15, 15))

    display_list = (img, mask)
    title = ['Input Image', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
    plt.savefig('predicted_mask.png')


def confusion_matrix_generate(y_true, y_pred, class_names, normalize=False):
    """ Generates a confusion matrix

    Args:
        y_true (list of str): list of true labels
        y_pred (list of str): list of predicted labels
        class_names (list of str): list of classe names
        normalize (bool): If True then the object returned will contain the relative frequencies of the unique values.

    Returns:
        conf_matrix: The confusion matrix
    """

    conf_matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    if normalize:
        conf_matrix = conf_matrix / conf_matrix.astype(np.float).sum(axis=1)

    sns.heatmap(conf_matrix, cbar=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot=True)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig('tests/confusion_matrix.png')

    return conf_matrix


def model_evaluate(model_path, input_path, img_height, img_width, num_channels, class_names=None):
    """

    Args:
        model_path (tf.h5): CNN model to be trained
        input_path (str): Path to the folder containing test images
        img_height (int): Image height
        img_width (int): Image width
        num_channels (int): Number of channels on the images
        class_names (list of str): List of classes

    Returns:

    """
    # Class names based on the directory structure
    class_names = os.listdir(input_path)

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

    conf_matrix = confusion_matrix_generate(y_true, y_pred, class_names)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(class_report).transpose().to_csv('classification_report.csv')
    class_report = classification_report(y_true, y_pred, output_dict=False)

    return test_scores, conf_matrix, class_report
