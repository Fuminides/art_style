import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--data_path', type=str, default='data', help='Path to the data')
parser.add_argument('--train_file', type=str, default='train.csv', help='Path to the train file')
parser.add_argument('--val_file', type=str, default='test.csv', help='Path to the val file')
parser.add_argument('--samples', type=float, default=1.0, help='Fraction of samples to use')
parser.add_argument('--model_path', type=str, help='Path to the model')
parser.add_argument('--semart_path', type=str, help='Path to the semart dataset')
parser.add_argument('--csvtrain', default='semart_train.csv', help='Training set data file')
parser.add_argument('--csvval', default='semart_val.csv', help='Dataset val data file')
parser.add_argument('--csvtest', default='semart_test.csv', help='Dataset test data file')
parser.add_argument('--dir_images', default='Images/')
parser.add_argument('--workers', default=0, type=int)

