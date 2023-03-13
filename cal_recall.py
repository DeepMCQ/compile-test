import torch
import numpy as np
from sklearn.datasets import load_svmlight_file


def main():
    x, y = load_svmlight_file("dump_query", zero_based=False)
    print(y)
    y = torch.from_numpy(y)
    with open("song_result.txt", "r") as fp:
        result = fp.readlines()
    result = torch.tensor([int(r) for r in result])
    print((y == result).float().mean())


if __name__ == "__main__":
    main()
