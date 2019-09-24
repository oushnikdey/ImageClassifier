import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--image_path',
                    dest='image_path',
                    help='Set image path')

parser.add_argument('--checkpoint',
                    dest='checkpoint',
                    default='./checkpoint.pth',
                    help='Set checkpoint path')


parser.add_argument('--category_names',
                    dest='category_names',
                    default="./cat_to_name.json",
                    type=str,
                    help='Set category names mapping path')

parser.add_argument('--top_k',
                    dest='top_k',
                    default=5,
                    type=int,
                    help='Set top K most likely classes')

parser.add_argument('--arch',
                    dest='arch',
                    default='vgg16',
                    help='Set architecture')

parser.add_argument('--gpu',
                    default = False,
                    dest='gpu',
                    help='Use GPU for training')

results = parser.parse_args()

def get_args():
    return results