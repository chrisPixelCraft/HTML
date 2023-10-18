import numpy as np
import matplotlib.pyplot as plt


def eout_value(s, theta):
    return 0.5 - 0.4 * s + 0.4 * s * np.abs(theta)


def generate_data(n):
    x = np.random.uniform(-1, 1, n)
    noise = np.random.choice([1, -1], n, p=[0.9, 0.1])
    y = np.sign(x) * noise
    # print((x, y))
    return x, y


def solve(n, x, y):
    index = np.argsort(x)
    x_sorted = x[index]
    y_sorted = y[index]
    ein_tmp = float("inf")
    theta_tmp = -1
    s_tmp = -1

    for i in range(n - 1):
        theta = (x_sorted[i] + x_sorted[i + 1]) / 2
        for s in [-1, 1]:
            y_tmp = s * np.sign(x_sorted - theta)
            ein = np.mean(y_sorted != y_tmp)

            if ein < ein_tmp:
                ein_tmp = ein
                theta_tmp = theta
                s_tmp = s
            elif ein == ein_tmp and s * theta < s_tmp * theta_tmp:
                theta_tmp = theta
                s_tmp = s

    return ein_tmp, s_tmp, theta_tmp


def main():
    n = 32
    times = 2000

    eins = np.array([])
    eouts = np.array([])

    for i in range(2000):
        (x, y) = generate_data(n)
        (ein_val, s, theta) = solve(n, x, y)
        eout_val = eout_value(s, theta)

        print((ein_val, eout_val))
        eins = np.append(eins, ein_val)
        # print(eins)
        eouts = np.append(eouts, eout_val)
        # print(eouts)

    print(eins)
    print(eouts)
    plt.scatter(eins, eouts, alpha=0.5)
    plt.xlabel("E_in(g)")
    plt.ylabel("E_out(g)")
    plt.savefig("decision_stump_p10.png")
    # plt.show()

    print("Median of E_in(g) - E_out(g): ", np.median(eins - eouts))
    return


if __name__ == "__main__":
    main()
