import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--dir', default='./tcdata/', type=str)
parser.add_argument('-d', '--train_data', default='./tcdata/train/', type=str)
parser.add_argument('-t', '--test_data', default='./tcdata/test/', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
parser.add_argument('--epochs', default=10, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR')
parser.add_argument('--img_size', default=512, type=int, metavar='N')

parser.add_argument('--resume', default='./result/train/', type=str, metavar='PATH')
parser.add_argument('--result', default='./result/test/', type=str)

args = parser.parse_args()
