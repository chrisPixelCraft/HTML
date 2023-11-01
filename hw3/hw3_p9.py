# hw3_p9

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


def solve_linear_regression(x, y):
    pseudo_inverse = np.linalg.pinv(x)
    # print(np.shape(pseudo_inverse))
    # print(np.shape(y))
    w_lin = pseudo_inverse @ y
    return w_lin


def main():
    x_test, y_test = generate_data(4096)
    # print(train_x, train_y)

    E_sqr_in_values = np.array([])

    for i in range(128):
        x_train, y_train = generate_data(256)
        w_lin = solve_linear_regression(x_train, y_train)
        print(w_lin)
        E_sqr_in = np.mean((x_train @ w_lin - y_train) ** 2)
        E_sqr_in_values = np.append(E_sqr_in_values, E_sqr_in)

    print(E_sqr_in_values)
    median = np.median(E_sqr_in_values)
    print(f"Median of E_in:  {median}")

    plt.hist(E_sqr_in_values, bins=20)
    plt.title("$E_{in}^{sqr}$_p9")
    plt.xlabel("$E_{in}^{sqr}$")
    plt.ylabel("N")
    plt.savefig("E_in_p9.png")
    return


if __name__ == "__main__":
    main()
