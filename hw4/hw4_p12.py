import numpy as np
import matplotlib.pyplot as plt
import random
from liblinear.liblinearutil import *


def load_data():
    return np.loadtxt("train.dat", dtype=np.float64)


def transform(X):
    transform_X = np.array([1 for _ in range(X.shape[0])])
    for i in range(X.shape[1]):
        transform_X = np.column_stack((transform_X, X[:, i]))
        for j in range(i, X.shape[1]):
            transform_X = np.column_stack((transform_X, X[:, i] * X[:, j]))
            for k in range(j, X.shape[1]):
                transform_X = np.column_stack(
                    (transform_X, X[:, i] * X[:, j] * X[:, k])
                )
    return transform_X


# def solve(train_X, train_y, validation_X, validation_y, lambda_value):
#     prob = problem(train_y, train_X)

#     C = 1 / (2 * lambda_value)
#     param = parameter("-s 0 -c {} -e 0.000001 -q".format(C))
#     m = train(prob, param)
#     p_label, p_acc, p_val = predict(validation_y, validation_X, m)

#     err = (100 - p_acc[0]) / 100
#     # find the model has the min E_aug
#     return err


def solve(X, y, lambda_value):
    prob = problem(y, X)

    C = 1 / (2 * lambda_value)
    param = parameter("-s 0 -c {} -e 0.000001 -v 5 -q".format(C))
    p_acc = train(prob, param)
    # p_label, p_acc, p_val = predict(validation_y, validation_X, m)

    err = (100 - p_acc) / 100
    # find the model has the min E_aug
    return err


def main():
    data = load_data()
    X = data[:, :-1]
    y = data[:, -1]
    X = transform(X)
    v_fold = 40  # each fold is 40 in p12
    # print(v_fold) # 40

    log_lambdas = np.array([-6, -4, -2, 0, 2])
    lambdas = np.array([1e-6, 1e-4, 1e-2, 1e0, 1e2])
    # lambdas = np.array([1e2, 1e0, 1e-2, 1e-4, 1e-6])
    # print(lambdas)

    lambdas_ans = np.array([])

    for i in range(128):
        np.random.seed(i)
        rst = np.array([])

        for j in range(lambdas.size):
            err_val = np.array([])

            # for k in range(5):
            #     validation_X = data[k * v_fold : k * v_fold + v_fold, :-1]
            #     validation_y = data[k * v_fold : k * v_fold + v_fold, -1]
            #     train_X = np.concatenate(
            #         [data[: k * v_fold, :-1], data[k * v_fold + v_fold :, :-1]]
            #     )
            #     train_y = np.concatenate(
            #         [data[: k * v_fold, -1], data[k * v_fold + v_fold :, -1]]
            #     )
            #     # print(train_X.shape, train_y.shape)
            #     # print(validation_X.shape, validation_y.shape)

            err_cv = solve(X, y, lambdas[j])
            # err_val = np.append(err_val, err)
            # print(err_val.shape)
            # err_cv = np.mean(err_val)
            rst = np.append(rst, err_cv)
        print(rst)

        tmp = float("inf")
        ans = float("-inf")
        for j in range(rst.size):
            # if rst[j] <= tmp and np.log10(lambdas[j]) >= ans:
            if rst[j] < tmp:
                tmp = rst[j]
                ans = np.log10(lambdas[j])
            elif rst[j] == tmp:
                if np.log10(lambdas[j]) > ans:
                    ans = np.log10(lambdas[j])
        print(ans)
        # print(tmp, ans)
        lambdas_ans = np.append(lambdas_ans, int(ans))

    # print(lambdas_ans)
    plt.hist(lambdas_ans, bins=20)
    plt.savefig("hw4_p12.png")
    return


if __name__ == "__main__":
    main()
