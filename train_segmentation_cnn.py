import argparse

from src.data_processing import data_preprocess_segmentation
from src.classifiers import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-model', '--model_path', help='Path to the keras model', default='unet_mobile_v2.h5')
    ap.add_argument('-i', '--input_path', help='Directory containing all images', default='oxford_iiit_pet:3.*.*')
    ap.add_argument('-e', '--epochs', help='Number of epochs during training', default=100)
    ap.add_argument('-height', '--img_height', help='Images height', default=128)
    ap.add_argument('-width', '--img_width', help='Images width', default=128)
    ap.add_argument('-b_size', '--batch_size', help='Size of batches during training', default=64)
    ap.add_argument('-test_b_size', '--test_batch_size', help='Size of batches during validation and test',
                    default=16)
    args = vars(ap.parse_args())

    train_batches, test_batches, num_training_steps, num_test_steps = data_preprocess_segmentation(args['input_path'],
                                                                                                   args['img_height'],
                                                                                                   args['img_width'],
                                                                                                   args['batch_size'],
                                                                                                   args[
                                                                                                       'test_batch_size'])

    train_model(args['model_path'],
                train_batches,
                test_batches,
                args['epochs'],
                num_training_steps,
                num_test_steps,
                segmentation=True)


if __name__ == '__main__':
    main()
