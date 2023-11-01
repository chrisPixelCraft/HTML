# hw3_p12

import numpy as np
import matplotlib.pyplot as plt


def generate_data(n):
    y = np.random.choice([-1, 1], size=n)

    x = np.zeros((n, 3))
    x[:, 0] = 1
    for i in range(n):
        if y[i] == 1:
            x[i, 1:], _ = np.random.multivariate_normal(
                [3, 2], [[0.4, 0], [0, 0.4]], 1
            ).T
        elif y[i] == -1:
            x[i, 1:], _ = np.random.multivariate_normal(
                [5, 0], [[0.6, 0], [0, 0.6]], 1
            ).T

    return x, y


def generate_outlier(n):
    y = np.ones(n)

    x = np.zeros((n, 3))
    x[:, 0] = 1
    for i in range(n):
        x[i, 1:], _ = np.random.multivariate_normal([0, 6], [[0.1, 0], [0, 0.3]], 1).T

    return x, y


def solve_linear_regression(x, y):
    pseudo_inverse = np.linalg.pinv(x)
    # print(np.shape(pseudo_inverse))
    # print(np.shape(y))
    w_lin = pseudo_inverse @ y
    return w_lin


def logistic_function(s):
    return 1 / (1 + np.exp(-s))


def solve_logistic_regression(x, y, eta, T):
    w_size = x.shape[1]
    w_t = np.zeros(x.shape[1])

    for i in range(T):
        # unit_gradient by np.linalg.norm
        E_gradient = (logistic_function(-x @ w_t @ y) * (-y @ x)) / w_size  # 3*1
        norm = np.linalg.norm(E_gradient)  # single norm value

        # you can't divide E_gradient by norm when E_gradient approaches to zero
        # (if E_gradient is 0, norm will be zero. However, you can't divide any number by zero)
        # we only do iteration when E_gradient is not 0. Instead, it will terminate
        if norm == 0:
            return w_t
        else:
            v = -1 * E_gradient / norm
            # print(v)
            w_t = w_t + eta * v
            # final E_gradient will be 0, which means we have minimum E_in with hypothesis w_t
            # print(w_t)
    return w_t


def main():
    eta = 0.1
    T = 500
    n_train = 256
    n_test = 4096
    n_outlier = 16

    E_out_A = np.array([])
    E_out_B = np.array([])

    for i in range(128):
        x_train, y_train = generate_data(n_train)
        x_test, y_test = generate_data(n_test)
        x_outlier, y_outlier = generate_outlier(n_outlier)

        x_train_with_outlier = np.concatenate((x_train, x_outlier), axis=0)
        y_train_with_outlier = np.concatenate((y_train, y_outlier), axis=0)
        # print(np.shape(x_train_with_outlier), np.shape(y_train_with_outlier))
        # print(x_train_with_outlier, y_train_with_outlier)
        # print(x_test, y_test)

        w_lin = solve_linear_regression(x_train_with_outlier, y_train_with_outlier)
        E_binary_classification_out_A = np.mean(np.sign(x_test @ w_lin) != y_test)
        E_out_A = np.append(E_out_A, E_binary_classification_out_A)

        w_log = solve_logistic_regression(
            x_train_with_outlier, y_train_with_outlier, eta, T
        )
        E_binary_classification_out_B = np.mean(np.sign(x_test @ w_log) != y_test)
        E_out_B = np.append(E_out_B, E_binary_classification_out_B)
        print(w_lin, w_log)

    plt.scatter(E_out_A, E_out_B, alpha=0.6)
    plt.xlabel("E_out of algorithm A (linear regression)")
    plt.ylabel("E_out of algorithm B (logistic regression)")
    plt.title("Scatter plot of E_out for algorithm A and B_p12")
    plt.savefig("E_out_p12.png")

    median_A = np.median(E_out_A)
    median_B = np.median(E_out_B)
    print(f"Median E_out for Algorithm A: {median_A}")
    print(f"Median E_out for Algorithm B: {median_B}")

    return


if __name__ == "__main__":
    main()
