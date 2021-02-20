import time
import matplotlib.pyplot as plt
import concurrent.futures
from util import *
import os

def problem1():
    maze = generate_maze(5, .2)
    print_maze(maze)


def problem2():
    # We use a process pool to submit many trials at once
    exec = concurrent.futures.ProcessPoolExecutor()
    # Our paramters to the simulation
    dim = 100
    trials_per_p = 80*32
    p_steps = 2000

    # Generating the datapoints
    x = [p/p_steps for p in range(p_steps)]
    y = [0 for _ in range(p_steps)]

    # Submits all our jobs 
    futures = {exec.submit(p2_trial, dim, p/p_steps) : p/p_steps for _ in range(trials_per_p) for p in range(p_steps)}
    
    # Process the data once the job completes
    for f in concurrent.futures.as_completed(futures):
        if f.result():
            y[round(futures[f] * p_steps)] += 1
    y = [i/trials_per_p for i in y]

    # Plot the data and save the graph
    plt.scatter(x, y, s=[1 for _ in x])
    plt.title('Figure 2')
    plt.xlabel('Obstacle Density p')
    plt.ylabel('Probability S reachable from G')
    plt.savefig('figure2.png')
    plt.show()
    

def p2_trial(dim, p):
    # Subroutine submitted as a single trial
    maze = generate_maze(dim, p)
    return dfs(maze, (0, 0), (dim - 1, dim - 1))

def problem3():
    # Process pool
    exec = concurrent.futures.ProcessPoolExecutor()
    # Parameters
    dim = 100
    trials_per_p = 600
    p_steps = 1000

    # Euclidean distance heuristic sent as map to optimize calculation
    h_map = {}
    h = lambda f, g : math.sqrt(math.pow(f[0]-g[0], 2) + math.pow(f[1]-g[1], 2))
    for i in range(dim):
        for j in range(dim):
            h_map[(i, j)] = h((i, j), (dim - 1, dim - 1))

    # Set up of data
    x = [p / p_steps for p in range(p_steps)]
    y = [0 for _ in range(p_steps)]

    # Submit all the jobs and process the info
    futures = {exec.submit(p3_trial, h_map, dim, p / p_steps): p / p_steps for _ in range(trials_per_p) for p in range(p_steps)}
    for f in concurrent.futures.as_completed(futures):
        y[round(futures[f] * p_steps)] += f.result()
    y = [i / trials_per_p for i in y]

    # Plot the data
    plt.scatter(x, y, s=[1 for _ in x])
    plt.title('Figure 3')
    plt.xlabel('Obstacle Density p')
    plt.ylabel('BFS - A* nodes explored')
    plt.savefig('figure3.png')
    plt.show()


def p3_trial(h_map, dim, p):
    # Subroutine to generate the difference in nodes
    maze = generate_maze(dim, p)
    return bfs(maze, (0, 0), (dim - 1, dim - 1)) - a_star(maze, (0, 0), (dim - 1, dim - 1), h_map = h_map)


def problem4():
    # Process pool
    exec = concurrent.futures.ProcessPoolExecutor()

    x, y = [], []

    # Submit jobs
    futures = {exec.submit(p4_trial, dim*1000, 0.3): dim*1000 for _ in range(100) for dim in range(1, 6)}

    # Process results
    for f in concurrent.futures.as_completed(futures):
        x.append(futures[f]*1000)
        y.append(f.result())

    # Plot and display
    plt.scatter(x, y, s=[1 for _ in x])
    plt.title('Figure 4c')
    plt.xlabel('Maze dimension')
    plt.ylabel('DFS Execution time (s)')
    plt.savefig('figure4.png')
    plt.show()


def p4_trial(dim, p):
    maze = generate_maze(dim, p)
    t = time.time()
    # Switch functions for different graphs of algorithms

    # Euclidean distance heuristic sent as map to optimize calculation
    # h_map = {}
    # h = lambda f, g : math.sqrt(math.pow(f[0]-g[0], 2) + math.pow(f[1]-g[1], 2))
    # for i in range(dim):
    #     for j in range(dim):
    #         h_map[(i, j)] = h((i, j), (dim - 1, dim - 1))
    # a_star(maze, (0, 0), (dim - 1, dim - 1), h_map = h_map)

    # bfs(maze, (0, 0), (dim - 1, dim - 1))

    dfs(maze, (0, 0), (dim - 1, dim - 1))

    return (time.time() - t)


def problem6():
    # Process pool
    exec = concurrent.futures.ProcessPoolExecutor()
    # Parameters
    q_steps = 7*3*2
    q_trials = 7*6*2
    dim = 100

    h_map = {}
    h = lambda f, g : math.sqrt(math.pow(f[0]-g[0], 2) + math.pow(f[1]-g[1], 2))
    h = lambda f, g : math.fabs(f[0]-g[0]) + math.fabs(f[1]-g[1])

    for i in range(dim):
        for j in range(dim):
            h_map[(i, j)] = h((i, j), (dim - 1, dim - 1))

    # Generate the data, shift x values by small amount to prevent overlap
    xs = [[i / 200 + q / q_steps for q in range(q_steps)] for i in range(3)]
    ys = [[0 for _ in range(q_steps)] for _ in range(3)]

    # Submit the jobs to the executor
    futures = {exec.submit(p6_trial, h_map = h_map, dim = dim, q = q / q_steps): q / q_steps for _ in range(q_trials) for q in range(q_steps)}

    # Process the completed jobs
    for f in concurrent.futures.as_completed(futures):
        results = f.result()
        for i in range(3):
            ys[i][round(futures[f] * q_steps)] += results[i] / q_trials

    # Plot the data
    plt.scatter(xs[0], ys[0], s=5, c="Red", label = "Strategy 1")
    plt.scatter(xs[1], ys[1], s=5, c="Blue", label = "Strategy 2")
    plt.scatter(xs[2], ys[2], s=5, c="Green", label = "Strategy 3")
    plt.legend()
    # Plot the graph
    plt.title('Figure 6')
    plt.xlabel('Flammability Rate, q')
    plt.ylabel('Success Rate')
    plt.savefig(dirname + 'figure6.png')
    plt.show()


def p6_trial(dim = 100, p = 0.3, q = 0.3, h_map = None, debug = False):
    results, nexts, deqs = [-1] * 3, [(0, 0)] * 3, [None] * 3

    # Generate a maze where start is reachable from goal and fire is reachable from start
    while not deqs[0] or not deqs[1]:
        maze = generate_maze(dim, p)
        fires = [start_fire(maze)]
        maze[fires[0][0]][fires[0][1]] = 0
        deqs[0] = a_star(maze, nexts[0], (len(maze) - 1, len(maze) - 1), h_map = h_map, path=True)
        deqs[1] = a_star(maze, nexts[1], fires[0], h_map = h_map, path=True)
        maze[fires[0][0]][fires[0][1]] = 2
        
    future_mazes = []
    round = 0
    while True:
        # Generate future fire mazes for our SCORCH algorithm
        if round % 4 == 0:
            future_mazes.clear()
            fires_2, maze_2 = fires[:], [r[:] for r in maze]
            for _ in range(2 * (dim - 1)):
                maze_2, fires_2 = sim_maze(maze_2, fires_2, q)
                future_mazes.append(maze_2)
        round += 1

        # Ask Strategy 2 and 3 for the next move given the particular info
        deqs[1] = a_star(maze, nexts[1], (len(maze) - 1, len(maze) - 1), h_map=h_map, path=True)
        # This is SCORCH
        deqs[2] = a_star(maze, nexts[2], (len(maze) - 1, len(maze) - 1), h_map=h_map, f = future_mazes, r = round % 4, path=True)

        # Simulate one step of the maze
        maze, fires = tick_maze(maze, fires, q)

        if debug:
            print_maze(maze, nexts[1])
            print (deqs[0], deqs[1])

        # For each strategy, move the agent and check if it is on fire
        for i in range(len(deqs)):
            if deqs[i]:
                nexts[i] = deqs[i].popleft()
                if maze[nexts[i][0]][nexts[i][1]] != 0:
                    results[i] = 0
                    deqs[i].clear()

        # If all of the agents are finished, then check which ones made it to goal node
        if not [deq for deq in deqs if deq]:
            for i in range(len(results)):
                results[i] = 1 if nexts[i] == (len(maze) - 1, len(maze) - 1) else 0
            break

        if debug:
            print (nexts[1])

    # Return the results
    return results


if __name__ == "__main__":
    problem1()
    problem2()
    problem3()
    problem4()
    problem6()
