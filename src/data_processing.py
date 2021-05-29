import os
import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_label(file_path, class_names):
    """ Gets the input label based on its folder

    Args:
        file_path (str): Path of the input image
        class_names (list of str):  List of strings containing each class of the input folders
    Returns:
        label (str): input images' label
    """
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    label = parts[-2] == class_names

    return label


def read_image(file_path, num_channels):
    """ Reads image

    Args:
        file_path (str): Path of the input image
        num_channels (int): Number of channels on the images

    Returns:
        img (tf.image): Image in tensor format
    """
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, num_channels)

    return img


def process_img(img, img_height, img_width):
    """ converts and resize images

    Args:
        img (tf.image): Input image
        img_height (int): Image height
        img_width (int): Image width
    Returns:
        img (tf.image): Image after preprocessing
    """
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.image.resize(img, [img_height, img_width])

    return img


def process_mask(img, img_height, img_width):
    """ converts and resize mask images

        Args:
            img (tf.image): Input image
            img_height (int): Image height
            img_width (int): Image width
        Returns:
            img (tf.image): Image mask after preprocessing
    """
    img = tf.image.resize(img, [img_height, img_width])
    # initial class marked as 0
    img -= 1

    return img


def load_images(file_path, class_names, img_height, img_width, num_channels):
    """ Preprocess images
    Args:
        file_path (str): Path of the input image
        class_names (list of str): List of strings containing each class of the input folders
        img_height (int): Image height
        img_width (int): Image width
        num_channels (int): Number of channels on the images
    Returns:
        img (tf.image): Input image
        label (str): Input images' label
    """
    # get label
    label = get_label(file_path, class_names)
    # read image
    img = read_image(file_path, num_channels)
    # process image
    img = process_img(img, img_height, img_width)

    return img, label


def load_images_segmentation(dataset, img_height, img_width):
    """

    Args:
        dataset (dict):
        img_height (int): Image height
        img_width (int): Image width

    Returns:
        input_image
        input_mask

    """
    input_image = process_img(dataset['image'], img_height, img_width)
    input_mask = process_mask(dataset['segmentation_mask'], img_height, img_width)

    return input_image, input_mask


def data_augmentation(img, label):
    """ Data Augmentation of training images

    Args:
        img (tf.image): Input image
        label (str): Input images' label

    Returns:
        img (tf.image): Input image after augmentation
        label (str): Input images' label
    """
    # Rotate
    img = tf.image.rot90(img, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # Flip
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    # Contrast
    img = tf.image.random_contrast(img, 0.8, 1.3)
    # Saturation
    img = tf.image.random_saturation(img, 0.6, 1.3)
    # Brightness
    img = tf.image.random_brightness(img, 0.1)

    return img, label


def prepare_for_training(ds, batch_size, augment, cache=True, shuffle_buffer_size=1000):
    """ Prepares the dataset for training

    Args:
        ds: Tensorflow.dataset -- Dataset of input images
        batch_size: int -- Number of images per batch
        augment (bool):
        cache: bool -- Saves the input dataset in memory
        shuffle_buffer_size: int -- Maximum number of elements that will be buffered when prefetching.
    Returns:
        ds: Tensorflow.dataset -- Training batches ready for training
    """
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    # Shuffles the training images
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()
    # Create training batches
    ds = ds.batch(batch_size)
    # Augment
    if augment:
        ds = ds.map(data_augmentation, num_parallel_calls=AUTOTUNE)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def data_preprocess(input_path, img_height, img_width, num_channels, batch_size, val_batch_size, augment=False,
                    class_names=None):
    """ Prepare the dataset for training validation and test

    Args:
        input_path (str): Path of the input image
        img_height (int): Image height
        img_width (int): Image width
        num_channels (int): Number of channels on the images
        batch_size (int): Number of images per batch
        val_batch_size (int): Number of images per batch in validation
        augment (bool):
        class_names (list of str):  List of strings containing each class of the input folders
    Returns:
        train_data (tf.Dataset): Batches of training images
        validation_batches (tf.Dataset): Batches of validation images
        validation_batches (tf.Dataset): Batches of test images
        num_training_steps (int): Number of training steps per epoch
        num_val_steps (int): Number of steps during validation
    """
    # Class names based on the directory structure
    class_names = os.listdir(input_path)

    train_dir = pathlib.Path(os.path.join(input_path, 'train_images'))
    val_dir = pathlib.Path(os.path.join(input_path, 'val_images'))

    train_image_count = len(list(train_dir.glob('*/*.jpg')))
    val_image_count = len(list(val_dir.glob('*/*.jpg')))

    num_training_steps = train_image_count // batch_size
    num_val_steps = val_image_count // val_batch_size

    train_ds = tf.data.Dataset.list_files(str(train_dir / '*/*'))
    val_ds = tf.data.Dataset.list_files(str(val_dir / '*/*'))

    train = train_ds.map(lambda file_path: load_images(file_path, class_names, img_height, img_width, num_channels),
                         num_parallel_calls=AUTOTUNE)
    validation = val_ds.map(lambda file_path: load_images(file_path, class_names, img_height, img_width, num_channels),
                            num_parallel_calls=AUTOTUNE)

    train_batches = prepare_for_training(train, batch_size, augment)
    val_batches = validation.batch(val_batch_size)

    return train_batches, val_batches, num_training_steps, num_val_steps


def data_preprocess_segmentation(input_path, img_height, img_width, batch_size, test_batch_size, tfrecord=True,
                                 augment=False):
    """ Prepare the dataset for training validation and test

    Args:
        input_path (str): Path of the input image
        img_height (int): Image height
        img_width (int): Image width
        batch_size (int): Number of images per batch
        test_batch_size (int): Number of images per batch in validation
        tfrecord (bool):
        augment (bool):
    Returns:
        train_data (tf.data.Dataset): Batches of training images
        validation_batches (tf..data.Dataset): Batches of validation images
        validation_batches (tf.data.Dataset): Batches of test images
        num_training_steps (int): Number of training steps per epoch
        num_val_steps (int): Number of steps during validation
    """

    if tfrecord:
        dataset, info = tfds.load(input_path, with_info=True)

        num_training_steps = info.splits['train'].num_examples
        num_test_steps = info.splits['test'].num_examples

    else:
        # Class names based on the directory structure
        class_names = os.listdir(input_path)

        # o nome das pastas não precisa ser esses : image e segmentation_mask
        train_images_dir = pathlib.Path(os.path.join(input_path, 'train_images', 'image'))
        train_masks_dir = pathlib.Path(os.path.join(input_path, 'train_images', 'segmentation_mask'))
        test_images_dir = pathlib.Path(os.path.join(input_path, 'test_images', 'image'))
        test_masks_dir = pathlib.Path(os.path.join(input_path, 'test_images', 'segmentation_mask'))
        # verificar se as classes se separam por pasta mesmo + adicionar a possibilidade de imagens .png
        train_image_count = len(list(train_images_dir.glob('*/*.jpg')))
        test_image_count = len(list(test_images_dir.glob('*/*.jpg')))

        num_training_steps = train_image_count // batch_size
        num_test_steps = test_image_count // test_batch_size

        # ainda falta criar outro dicionário pras separçaões
        train_images = tf.data.Dataset.list_files(str(train_images_dir / '*/*'))
        train_masks = tf.data.Dataset.list_files(str(train_masks_dir / '*/*'))
        train_ds = {'image': train_images, 'segmentation_mask': train_masks}

        test_images = tf.data.Dataset.list_files(str(test_images_dir / '*/*'))
        test_masks = tf.data.Dataset.list_files(str(test_masks_dir / '*/*'))
        test_ds = {'image': test_images, 'segmentation_mask': test_masks}

        dataset = {'train': train_ds, 'test': test_ds}

    train = dataset['train'].map(lambda input_images: load_images_segmentation(input_images, img_height, img_width),
                                 num_parallel_calls=AUTOTUNE)
    test = dataset['test'].map(lambda input_images: load_images_segmentation(input_images, img_height, img_width))

    train_batches = prepare_for_training(train, batch_size, augment)
    test_batches = test.batch(test_batch_size)

    return train_batches, test_batches, num_training_steps, num_test_steps


def data_preprocess_test(test_path, img_height, img_width, num_channels, class_names=None):
    """ Prepare the dataset for training validation and test

    Args:
        test_path (str): Path of the input image
        img_height (int): Image height
        img_width (int): Image width
        num_channels (int): Number of channels on the images
        class_names (list of str): List of classes
    Returns:
        test_data (tf.Dataset): Batches of test images
    """
    # Class names based on the directory structure
    class_names = os.listdir(test_path)

    test_dir = pathlib.Path(os.path.join(os.getcwd(), test_path))
    test_ds = tf.data.Dataset.list_files(str(test_dir / '*/*'))
    test_data = test_ds.map(lambda file_path: load_images(file_path, class_names, img_width, img_height, num_channels),
                            num_parallel_calls=AUTOTUNE)
    test_data = test_data.cache().batch(16)

    return test_data
