import numpy as np
import matplotlib.pyplot as plt
import random
import pandas


def load_data():
    return np.loadtxt("data.dat", dtype=np.float64)


def sampling(data, x_0):
    sample = random.choice(data)
    sample = np.insert(sample, 0, x_0)
    x = sample[:-1]
    y = sample[-1]
    return (x, y)


def is_Mistake(w, x, y):
    if y * np.dot(w, x) < 0:
        return True
    elif y * np.dot(w, x) == 0 and y > 0:
        return True
    return False


def pla_Solve(x_0, rst):
    cnt = 0  # update number for current train
    data = load_data()
    w = np.zeros(13)
    (x, y) = sampling(data, x_0)

    tmp = 0
    while tmp <= 5 * data.shape[0]:
        if is_Mistake(w, x, y):
            w += y * x
            tmp = 0
            cnt += 1
        else:
            tmp += 1
            (x, y) = sampling(data, x_0)
    print(w)
    rst.append(cnt)
    return


def main():
    rst = []  # update number for mistake correction times
    x_0 = 1
    for i in range(1000):
        pla_Solve(x_0, rst)

    # output
    print(rst)
    print(np.median(rst))
    plt.hist(rst, bins=200)
    plt.savefig("PLA_P9.png")
    return


if __name__ == "__main__":
    main()
