import os
import numpy as np
import argparse
from util import set_seed

def make_train_valid_test(filename, output_path, train_ratio, valid_ratio, test_ratio):
    with open(filename, 'r', encoding='utf-8') as fp:
        triple_list = [x for x in fp.readlines()]

    def get_split_index():
        ratio_sum = train_ratio + valid_ratio + test_ratio
        one = train_ratio * len(triple_list) / ratio_sum
        two = (train_ratio + valid_ratio) * len(triple_list) / ratio_sum
        return int(one), int(two)

    def save_triple_set(data, save_file):
        with open(save_file, 'w', encoding='utf-8') as fp:
            for x in data:
                fp.write(x)

    np.random.shuffle(triple_list)
    point_1, point_2 = get_split_index()
    train_list = triple_list[:point_1]
    valid_list = triple_list[point_1:point_2]
    test_list  = triple_list[point_2:]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    save_triple_set(train_list, os.path.join(output_path, 'train_triple_id.txt'))
    save_triple_set(valid_list, os.path.join(output_path, 'valid_triple_id.txt'))
    save_triple_set(test_list,  os.path.join(output_path, 'test_triple_id.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2020,
                        help="seed")
    parser.add_argument("--triple_file", type=str,
                        help="triple file path")
    parser.add_argument("--train_ratio", type=float, default=0.6,
                        help="train ratio")
    parser.add_argument("--valid_ratio", type=float, default=0.2,
                        help="valid ratio")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="test ratio")
    parser.add_argument("--save_path", type=str,
                        help="output files path")

    args = parser.parse_args()

    set_seed(args.seed)

    make_train_valid_test(args.triple_file, args.save_path, args.train_ratio, args.valid_ratio, args.test_ratio)