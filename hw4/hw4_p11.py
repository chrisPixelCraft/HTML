import numpy as np
import matplotlib.pyplot as plt
from liblinear.liblinearutil import *


def load_data():
    return np.loadtxt("train.dat", dtype=np.float64)


def solve(train_X, train_y, validation_X, validation_y, lambdas):
    prob = problem(train_y, train_X)
    acc = np.array([])

    for i in range(lambdas.size):
        C = 1 / (2 * lambdas[i])
        param = parameter("-s 0 -c {} -e 0.000001".format(C))
        m = train(prob, param)
        p_label, p_acc, p_val = predict(validation_y, validation_X, m)
        acc = np.append(acc, p_acc[0])
    print(acc)

    # find the model has the min E_aug
    tmp = 1e6
    ans = 1e6
    for i in range(lambdas.size):
        if (100 - acc[i]) / 100 <= tmp:
            tmp = (100 - acc[i]) / 100
            ans = np.log10(lambdas[i])
            print(tmp, ans)
    return tmp, ans


def main():
    data = load_data()
    log_lambdas = np.array([-6, -4, -2, 0, 2])
    lambdas = np.array([1e-6, 1e-4, 1e-2, 1e0, 1e2])
    # print(lambdas)
    rst = np.array([])

    for i in range(128):
        np.random.shuffle(data)
        train_X = data[:120, :-1]
        train_y = data[:120, -1]
        validation_X = data[120:200, :-1]
        validation_y = data[120:200, -1]
        err, ans = solve(train_X, train_y, validation_X, validation_y, lambdas)
        rst = np.append(rst, ans)

    print(rst)
    plt.hist(rst, bins=20)
    plt.savefig("hw4_p11.png")
    return


if __name__ == "__main__":
    main()
