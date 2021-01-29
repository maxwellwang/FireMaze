import time, random
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
    trials_per_p = 60
    p_steps = 2000

    t = time.time()

    x = [p / p_steps for p in range(p_steps)]
    y = [0 for _ in range(p_steps)]
    for k in range(10):
        futures = {exec.submit(p3_trial, dim, p / p_steps): p / p_steps for _ in range(trials_per_p) for p in range(p_steps//2)}
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

def problem6():
    debug = True
    base_maze = generate_maze(8, 0.2)
    base_fires = [start_fire(base_maze)]


    maze, fires = deepcopy(base_maze), deepcopy(base_fires)
    results = [-1] * 3
    nexts = [(0, 0)] * 3
    deqs = [a_star(maze, nexts[0], (len(maze) - 1, len(maze) - 1), path=True), None, None]
    while True:
        if debug:
            print_maze(maze, nexts[1])
        print (deqs[0], deqs[1])
        if deqs[0]:
            nexts[0] = deqs[0].popleft()
            if maze[nexts[0][0]][nexts[0][1]] != 0:
                results[0] = 0
                deqs[0].clear()

        deqs[1] = a_star(maze, nexts[1], (len(maze) - 1, len(maze) - 1), path=True)
        if deqs[1]:
            nexts[1] = deqs[1].popleft()

        if not deqs[0] and not deqs[1]:
            results[0] = 1 if nexts[0] == (len(maze) - 1, len(maze) - 1) else 0
            results[1] = 1 if nexts[1] == (len(maze) - 1, len(maze) - 1) else 0
            break

        maze, fires = tick_maze(maze, fires, 0.4)
        if debug:
            print (nexts[1])

    print (results)
    return results



    # maze, fires = deepcopy(base_maze), deepcopy(base_fires)
    # next = (0, 0)
    # result_two = 0
    # while next:
    #     if debug:
    #         print_maze(maze, next)
    #     deq = a_star(maze, next, (len(maze) - 1, len(maze) - 1), path=True)
    #     if deq:
    #         next = deq.popleft()
    #     else:
    #         if next == (len(maze) - 1, len(maze) - 1):
    #             result_two = Tru1
    #         break
    #     if debug:
    #         print(next)
    #     maze, fires = tick_maze(maze, fires, 0.3)


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
