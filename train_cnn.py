import argparse

from src.data_processing import data_preprocess
from src.classifiers import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-model', '--model_path', help='Path to the keras model', default='model.h5')
    ap.add_argument('-i', '--input_path', help='Directory containing all images', default='data/x')
    ap.add_argument('-e', '--epochs', help='Number of epochs during training', default=100)
    ap.add_argument('-height', '--img_height', help='Images height', default=336)
    ap.add_argument('-width', '--img_width', help='Images width', default=336)
    ap.add_argument('-channels', '--num_channels', help='Number of channels on the images', default=3)
    ap.add_argument('-batch_size', '--batch_size', help='Size of batches during training', default=32)
    ap.add_argument('-val_batch_size', '--val_batch_size', help='Size of batches during validation and test', default=16)
    ap.add_argument('-classes', '--class_names', help='List of classes', default=['class1',
                                                                                  'class2'])
    args = vars(ap.parse_args())

    train_batches, val_batches, num_training_steps, num_val_steps = data_preprocess(args['input_path'],
                                                                                    args['img_height'],
                                                                                    args['img_width'],
                                                                                    args['num_channels'],
                                                                                    args['batch_size'],
                                                                                    args['val_batch_size'],
                                                                                    args['class_names'])

    train_model(args['model_path'], train_batches, args['epochs'], val_batches, num_training_steps, num_val_steps)


if __name__ == '__main__':
    main()
