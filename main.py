import time
import random
import matplotlib.pyplot as plt
import concurrent.futures
from copy import deepcopy
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

    x = [p / p_steps for p in range(p_steps)]
    y = [0 for _ in range(p_steps)]

    futures = {exec.submit(p2_trial, dim, p / p_steps): p / p_steps for _ in range(trials_per_p) for p in
               range(p_steps)}
    for f in concurrent.futures.as_completed(futures):
        if f.result():
            y[round(futures[f] * p_steps)] += 1
    y = [i / trials_per_p for i in y]

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
    trials_per_p = 60
    p_steps = 2000

    t = time.time()

    x = [p / p_steps for p in range(p_steps)]
    y = [0 for _ in range(p_steps)]
    for k in range(10):
        futures = {exec.submit(p3_trial, dim, p / p_steps): p / p_steps for _ in range(trials_per_p) for p in
                   range(p_steps // 2)}
        for f in concurrent.futures.as_completed(futures):
            y[round(futures[f] * p_steps)] += f.result()
    y = [i / trials_per_p / k for i in y]

    print(time.time() - t)

    # plt.plot(x, y)
    plt.scatter(x, y, s=[1 for _ in x])
    plt.title('Figure 3')
    plt.xlabel('Obstacle Density p')
    plt.ylabel('BFS - A* nodes explored')
    plt.savefig('figure3.png')
    plt.show()


def p3_trial(dim, p):
    maze = generate_maze(dim, p)
    return bfs(maze, (0, 0), (dim - 1, dim - 1)) - a_star(maze, (0, 0), (dim - 1, dim - 1))


def problem4():
    exec = concurrent.futures.ProcessPoolExecutor()

    t = time.time()

    x, y = [], []
    futures = {exec.submit(p4_trial, dim * 1000, 0.3): dim * 1000 for _ in range(100) for dim in range(1, 5)}
    for f in concurrent.futures.as_completed(futures):
        x.append(futures[f] * 1000)
        y.append(f.result())

    print(time.time() - t)

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
    return time.time() - t


def problem6():
    exec = concurrent.futures.ProcessPoolExecutor()
    q_steps = 40
    q_trials = 40
    dim = 100

    t = time.time()

    x = [q / q_steps for q in range(q_steps)]
    ys = [[0 for _ in range(q_steps)] for _ in range(3)]
    h_map = {}
    h = lambda f, g: math.sqrt(math.pow(f[0] - g[0], 2) + math.pow(f[1] - g[1], 2))
    for i in range(dim):
        for j in range(dim):
            h_map[(i, j)] = h((i, j), (dim - 1, dim - 1))

    futures = {exec.submit(p6_trial, h_map=h_map, dim=dim, q=q / q_steps): q / q_steps for _ in range(q_trials) for q in
               range(q_steps)}
    for f in concurrent.futures.as_completed(futures):
        results = f.result()
        for i in range(3):
            ys[i][round(futures[f] * q_steps)] += results[i] / q_trials

    print(time.time() - t)

    plt.scatter(x, ys[0], s=[5 for _ in x], c="Red")
    plt.scatter(x, ys[1], s=[5 for _ in x], c="Blue")
    plt.scatter(x, ys[2], s=[5 for _ in x], c="Green")
    plt.title('Figure 6')
    plt.xlabel('Flammability Rate, q')
    plt.ylabel('Success Rate')
    plt.savefig('figure6.png')
    plt.show()


def p6_trial(dim=100, p=0.3, q=0.3, h_map=None, debug=False):
    results, nexts, deqs = [-1] * 3, [(0, 0)] * 3, [None] * 3

    maze = generate_maze(dim, p)
    fires = [start_fire(maze)]
    maze[fires[0][0]][fires[0][1]] = 2
    while not dfs(maze, (0, 0), (dim - 1, dim - 1)) or not dfs(maze, (0, 0), (fires[0][0], fires[0][1])):
        maze = generate_maze(dim, p)
        fires = [start_fire(maze)]
        maze[fires[0][0]][fires[0][1]] = 2

    deqs[0] = a_star(maze, nexts[0], (dim - 1, dim - 1), h_map=h_map, path=True)
    deqs[1] = a_star(maze, nexts[1], (dim - 1, dim - 1), h_map=h_map, path=True)
    deqs[2] = prune(maze, nexts[2], (dim - 1, dim - 1), h_map, fires, q)
    while True:
        deqs[1] = a_star(maze, nexts[1], (dim - 1, dim - 1), h_map=h_map, path=True)

        if debug:
            print_maze(maze, nexts[1])
            print(deqs[0], deqs[1], deqs[2])

        for i in range(len(deqs)):
            if deqs[i]:
                nexts[i] = deqs[i].popleft()
                if maze[nexts[i][0]][nexts[i][1]] != 0:
                    results[i] = 0
                    deqs[i].clear()
        maze, fires = tick_maze(maze, fires, q)

        if not [deq for deq in deqs if deq]:
            for i in range(len(results)):
                results[i] = 1 if nexts[i] == (dim - 1, dim - 1) else 0
            break

        if debug:
            print(nexts[1])
    print(results)
    return results


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
    # problem3()
    # problem4()
    # fire_sim()
    problem6()
