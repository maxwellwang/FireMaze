from util import *
import matplotlib.pyplot as plt
import concurrent.futures
import time
import random

def problem1():
    maze = generate_maze(5, .2)
    print_maze(maze)


def problem2():
    exec = concurrent.futures.ProcessPoolExecutor()
    dim = 1000
    trials_per_p = 160
    p_steps = 800

    t = time.time()

    x = [p/p_steps for p in range(p_steps)]
    y = [0 for _ in range(p_steps)]
    futures = {exec.submit(p2_trial, dim, p/p_steps) : p/p_steps for _ in range(trials_per_p) for p in range(p_steps)}
    for f in concurrent.futures.as_completed(futures):
        if f.result():
            y[round(futures[f]*p_steps)] += 1
    y = [i/trials_per_p for i in y]

    print(time.time() - t)

    plt.plot(x, y)
    plt.title('Figure 2')
    plt.xlabel('Obstacle Density p')
    plt.ylabel('Probability S reachable from G')
    plt.savefig('figure2.png')
    plt.show()

def p2_trial(dim, p):
    maze = generate_maze(dim, p)
    return check_reachable(maze, (0, 0), (dim - 1, dim - 1))

# problem1()
if __name__ == "__main__":
    t = time.time()
    problem2()
    print (time.time() - t)