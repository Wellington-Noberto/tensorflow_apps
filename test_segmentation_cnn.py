import argparse

from src.data_utils import plot_segmentation_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-model', '--model_path', help='Path to the keras model', default='new_unet.h5')
    ap.add_argument('-i', '--input_image', help='Input image', default='data/test.jpg')
    ap.add_argument('-height', '--img_height', help='Images height', default=128)
    ap.add_argument('-width', '--img_width', help='Images width', default=128)
    ap.add_argument('-channels', '--num_channels', help='Images width', default=3)
    args = vars(ap.parse_args())

    # Process input image
    plot_segmentation_mask(args['model_path'],
                           args['input_image'],
                           args['img_height'],
                           args['img_width'],
                           args['num_channels'])


if __name__ == '__main__':
    main()
