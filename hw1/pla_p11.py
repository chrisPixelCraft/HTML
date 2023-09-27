import numpy as np
import matplotlib.pyplot as plt
import random
import pandas


def get_data():
    data = np.loadtxt("data.dat", dtype=np.float64)
    return data


def check_mistake(w, x, y):
    if y * (np.dot(w, x)) < 0:
        return True
    elif y * (np.dot(w, x)) == 0:
        return True
    else:
        return False


def sampling(data, x_0):
    sample = random.choice(data)
    sample = np.insert(sample, 0, x_0)
    x = sample[0:-1]
    y = sample[-1]
    return (sample, x, y)


def solve():
    rst = []  # update number for mistake correction times
    for i in range(1000):
        x_0 = 11.26
        w = np.zeros(13)
        cnt = 0  # update number for current train
        data = get_data()
        (sample, x, y) = sampling(data, x_0)
        # print(data)
        # print(sample)

        tmp = 0
        while tmp <= 5 * data.shape[0]:
            if check_mistake(w, x, y):
                w += y * x
                cnt += 1
                tmp = 0
            else:
                tmp += 1
            (sample, x, y) = sampling(data, x_0)
        print(w)
        rst.append(cnt)

    # output
    print(rst)
    print(np.median(rst))
    plt.hist(rst, bins=200)
    # plt.show()
    plt.savefig("PLA_P11.png")


if __name__ == "__main__":
    solve()


# def solve(data):
