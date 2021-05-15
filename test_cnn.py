import argparse

from src.data_utils import model_evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-model', '--model_path', help='Path to the keras model', default='keratoDetect_model_classe.h5')
    ap.add_argument('-test_images', '--test_images', help='Path to test images', default='test_images')
    ap.add_argument('-height', '--img_height', help='Images height', default=336)
    ap.add_argument('-width', '--img_width', help='Images width', default=336)
    ap.add_argument('-channels', '--num_channels', help='Number of channels on the images', default=3)
    ap.add_argument('-classes', '--class_names', help='List of classes', default=['class1',
                                                                                  'class2'])

    args = vars(ap.parse_args())

    test_scores, conf_matrix, class_report = model_evaluate(args['model_path'],
                                                            args['test_images'],
                                                            args['img_height'],
                                                            args['img_width'],
                                                            args['num_channels'],
                                                            args['class_names'])

    print('Accuracy: ', test_scores[1])
    print('Loss: ', test_scores[0])
    print(conf_matrix)
    print(class_report)


if __name__ == "__main__":
    main()
