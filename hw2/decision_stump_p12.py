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
    s = np.random.choice([-1, 1])
    theta = np.random.uniform(-1, 1)
    # print(s, theta)
    eout = eout_value(s, theta)

    y_tmp = s * np.sign(x - theta)
    ein = np.mean(y != y_tmp)

    return ein, s, theta


def main():
    n = 8
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
    plt.savefig("decision_stump_p12.png")
    # plt.show()

    print("Median of E_in(g) - E_out(g): ", np.median(eins - eouts))
    return


if __name__ == "__main__":
    main()
