from util import *
import matplotlib.pyplot as plt


def problem1():
    maze = generate_maze(5, .2)
    print_maze(maze)


def problem2():
    dim = 1000
    trials_per_p = 100
    p_increment = .05
    p = p_increment
    x = []
    y = []
    while p < 1:
        x.append(p)
        sum = 0.0
        for i in range(trials_per_p):
            print('p = ' + str(p) + ', trial = ' + str(i + 1))
            maze = generate_maze(dim, p)
            sum += 1 if check_reachable(maze, (0, 0), (dim - 1, dim - 1)) else 0
        y.append(sum / trials_per_p)
        p += p_increment
    plt.plot(x, y)
    plt.title('Figure 2')
    plt.xlabel('Obstacle Density p')
    plt.ylabel('Probability That S Can Be Reached From G')
    plt.savefig('figure2.png')
    plt.show()


# problem1()
problem2()
