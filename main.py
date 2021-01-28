import time, random
import matplotlib.pyplot as plt
import concurrent.futures
from util import *


def problem1():
    maze = generate_maze(5, .2)
    print_maze(maze)


def problem2():
    exec = concurrent.futures.ProcessPoolExecutor()
    dim = 75
    trials_per_p = 80
    p_steps = 400

    t = time.time()

    x = [p/p_steps for p in range(p_steps)]
    y = [0 for _ in range(p_steps)]

    futures = {exec.submit(p2_trial, dim, p/p_steps) : p/p_steps for _ in range(trials_per_p) for p in range(p_steps)}
    for f in concurrent.futures.as_completed(futures):
        if f.result():
            y[round(futures[f] * p_steps)] += 1
    y = [i/trials_per_p for i in y]

    print(time.time() - t)

    # plt.plot(x, y)
    plt.scatter(x, y, s=[1 for _ in x])
    plt.title('Figure 2')
    plt.xlabel('Obstacle Density p')
    plt.ylabel('Probability S reachable from G')
    plt.savefig('figure2.png')
    plt.show()


def p2_trial(dim, p):
    maze = generate_maze(dim, p)
    return dfs(maze, (0, 0), (dim - 1, dim - 1))

def problem3():
    exec = concurrent.futures.ProcessPoolExecutor()
    dim = 75
    trials_per_p = 80
    p_steps = 200

    t = time.time()

    x = [p / p_steps for p in range(p_steps)]
    y = [0 for _ in range(p_steps)]

    futures = {exec.submit(p3_trial, dim, p / p_steps): p / p_steps for _ in range(trials_per_p) for p in range(p_steps)}
    for f in concurrent.futures.as_completed(futures):
        y[round(futures[f] * p_steps)] += f.result()
    y = [i / trials_per_p for i in y]

    print(time.time() - t)

    # plt.plot(x, y)
    plt.scatter(x, y, s=[1 for _ in x])
    plt.title('Figure 2')
    plt.xlabel('Obstacle Density p')
    plt.ylabel('BFS - A* nodes explored')
    plt.savefig('figure2.png')
    plt.show()


def p3_trial(dim, p):
    maze = generate_maze(dim, p)
    return bfs(maze, (0, 0), (dim - 1, dim - 1)) - a_star(maze, (0, 0), (dim - 1, dim - 1))


def problem4():
    exec = concurrent.futures.ProcessPoolExecutor()

    t = time.time()

    x, y = [], []
    futures = {exec.submit(p4_trial, dim*1000, 0.3): dim*1000 for _ in range(100) for dim in range(1, 5)}
    for f in concurrent.futures.as_completed(futures):
        x.append(futures[f]*1000)
        y.append(f.result())

    print (time.time() - t)

    # plt.plot(x, y)
    plt.scatter(x, y, s=[1 for _ in x])
    plt.title('Figure 4')
    plt.xlabel('Maze dimension')
    plt.ylabel('Execution time')
    plt.savefig('figure4.png')
    plt.show()


def p4_trial(dim, p):
    maze = generate_maze(dim, p)
    t = time.time()
    a_star(maze, (0, 0), (dim - 1, dim - 1))
    return (time.time() - t)


def fire_sim():
    maze = generate_maze(4, 0.3)
    fires = [start_fire(maze)]
    for _ in range(2):
        print_maze(maze)
        print()
        maze, fires = tick_maze(maze, fires, 0.9)

if __name__ == "__main__":
    # problem1()
    # problem2()
    problem3()
    # problem4()
    # fire_sim()
