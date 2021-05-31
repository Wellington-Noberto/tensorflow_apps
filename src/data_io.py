import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL = 'folder'
PATH = 'path'


def split_data(input_path, train_size, test_size):
    """ Create dataframes of the each split of data
    Args:
        input_path (str): Directory containing all images
        train_size (float): Proportion of dataset used for training
        test_size (float): Proportion of dataset used for test
    Returns:
        train_df, val_df, test_df (pandas.DataFrame)
    """

    files = []
    for dirpath, dirnames, filenames in os.walk(input_path):
        for path in filenames:
            file = os.path.join(dirpath, path)
            files.append(file)

    df = pd.DataFrame(files, columns=[PATH])
    df[LABEL] = df.path.str.split(os.sep).str.get(1)

    x_train = df.path
    y_train = df.folder

    train_size_int = int(train_size * x_train.count())
    test_size_int = int(test_size * x_train.count())

    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, train_size=train_size_int, random_state=42,
                                                        stratify=y_train)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=test_size_int, random_state=42,
                                                    stratify=y_test)

    train_df = df.iloc[X_train.index]
    val_df = df.iloc[X_val.index]
    test_df = df.iloc[X_test.index]

    train_df.to_csv('train_df.csv', index=False)
    val_df.to_csv('val_df.csv', index=False)
    test_df.to_csv('test_df.csv', index=False)

    return train_df, val_df, test_df


def split_folders(train_df, val_df, test_df, dir_path=''):
    """ Separate the dataset in folders for each split
    Args:
        train_df (pandas.DataFrame): Dataframe that contains the path and label of each image of the training split
        val_df (pandas.DataFrame): Dataframe that contains the path and label of each image of the validation split
        test_df (pandas.DataFrame): Dataframe that contains the path and label of each image of the test split
        dir_path (str): base directory name where the folders should be stored
    """

    split = (train_df, val_df, test_df)
    class_names = train_df[LABEL].unique()
    split_names = ('train_images', 'val_images', 'test_images')
    for split, split_name in zip(split, split_names):
        # Split directory
        split_path = os.path.join(dir_path, split_name)
        for class_name in class_names:
            # Class directory
            class_path = os.path.join(split_path, class_name)
            # Make directory
            if not os.path.exists(class_path):
                os.makedirs(class_path)

            split_df = split[split[LABEL] == class_name]
            class_images = (split_df[PATH]).values
            for file in class_images:
                shutil.move(file, class_path)


def split_images(input_path, train_size, test_size, dir_path):
    """
    Args:
        input_path (str): Directory containing all images
        train_size (float): Proportion of dataset used for training
        test_size (float): Proportion of dataset used for test
        dir_path (str): Base directory name where the folders should be stored
    Returns:
    """

    train_df, val_df, test_df = split_data(input_path, train_size, test_size)
    split_folders(train_df, val_df, test_df, dir_path)