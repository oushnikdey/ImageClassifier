import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir',
                    dest='data_dir',
                    default='./flowers',
                    help='Set data dir')

parser.add_argument('--save_dir',
                    dest='save_dir',
                    default='./checkpoint.pth',
                    help='Set checkpoint dir')

parser.add_argument('--arch',
                    dest='arch',
                    default='vgg16',
                    help='Set architecture')

parser.add_argument('--learning_rate',
                    dest='learning_rate',
                    default=0.001,
                    type=float,
                    help='Set learning rate')

parser.add_argument('--hidden_layers',
                    dest='hidden_layers',
                    default="1024, 256",
                    type=str,
                    help='Set hidden layers numbers. Separate each value by a comma')

parser.add_argument('--epochs',
                    dest='epochs',
                    default=4,
                    type=int,
                    help='Set epochs')

parser.add_argument('--gpu',
                    default = False,
                    dest='gpu',
                    help='Use GPU for training')

results = parser.parse_args()

def get_args():
    return results