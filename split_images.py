import argparse

from src.data_io import split_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input_path', help='Directory containing all images', default='data/raw/')
    ap.add_argument('-train', '--train_size', help='Proportion of dataset used for training', default=0.7)
    ap.add_argument('-test', '--test_size', help='Proportion of dataset used for test', default=0.15)
    ap.add_argument('-o', '--dir_path', help='Directory where the split should be stored', default='data/processed/')
    args = vars(ap.parse_args())

    split_images(args['input_path'], args['train_size'], args['test_size'], args['dir_path'])


if __name__ == "__main__":
    main()
