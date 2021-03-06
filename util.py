import heapq
import math
import random
from collections import deque
from copy import deepcopy

'''
Representation:
- maze is 2d list of integers 0-2
- 0 means empty
- 1 means occupied by obstacle
- 2 means on fire
- 3 means agent
'''

"""
Helper lambda function to check cell is valid
:param x: x coordinate of cell
:param y: y coordinate of cell
:param maze: The 2d array to check the particular cell on
:param val: Value that a cell should be to be valid, default 0
:return: Boolean of validity of cell
"""
valid_cell = lambda x, y, maze, val=0: 0 <= x < len(maze) and 0 <= y < len(maze) and maze[x][y] == val


def adj_cells(pair):
    """
    Helper function to return cells adjacent to input
    :param pair: Coordinates of some cell
    :return: Generator which produces the four adjacent coordinates
    """
    dx, dy = [0, -1, 0, 1], [-1, 0, 1, 0]
    for i in range(4):
        yield pair[0] + dx[i], pair[1] + dy[i]


def print_maze(maze, agent=None):
    """
    :param maze: Maze to be printed
    :return: None
    """

    if agent:
        maze[agent[0]][agent[1]] = 3

    symbols = {
        0: ".",
        1: "X",
        2: "F",
        3: "A"
    }

    for row in maze:
        print(''.join(symbols[x] for x in row))

    if agent:
        maze[agent[0]][agent[1]] = 0


def generate_maze(dim, p):
    """
    :param dim: Length/width of maze (square)
    :param p: Obstacle density, 0 <= p <= 1
    :return: Maze, dim by dim 2d array, cells either 0 or 1
    """

    # Validate dim
    if dim <= 0 or not isinstance(dim, int):
        return
    # Validate p
    if not 0 <= p <= 1:
        return

    # Generate maze object, spawning obstacles with probability p
    maze = [[1 if random.random() < p else 0 for _ in range(dim)] for _ in range(dim)]
    # Ensure top left and bottom right corners are open
    maze[0][0], maze[-1][-1] = 0, 0

    return maze


def start_fire(maze):
    """
    Modifies input maze in place and starts a fire
    :param maze: 2d array representing a maze
    :return: Tuple containing fire's coordinates
    """

    # Pick a random cell to spawn fire in
    dim = len(maze)
    cell = random.randint(1, dim * dim - 2)
    # Ensure that cell is open (no obstacles)
    while maze[cell // dim][cell % dim] != 0:
        cell = random.randint(1, dim * dim - 2)
    # Update cell with fire, then return
    maze[cell // dim][cell % dim] = 2
    return cell // dim, cell % dim


def tick_maze(maze, fires, q):
    """
    Spreads the fire one step in some maze
    :param maze: The maze to spread the fire in
    :param fires: A set containing coordinates of every fire
    :param q: The flammability rate of the fire
    :return: Tuple containing the new maze and new set of fires
    """

    def count_fires(maze, fire):
        """
        Helper function to count fires surronding a cell
        :param maze: Maze to count from
        :param fire: Pair of coordinates of interest
        :return: Number of fires surronding cell
        """
        num_fires = 0
        for dx, dy in adj_cells(fire):
            if valid_cell(dx, dy, maze, val=2):
                num_fires += 1
        return num_fires

    # Generate a copy of current fires and maze
    new_fires = fires[:]
    new_maze = [row[:] for row in maze]
    visited = set()

    # For each fire, we check its neighbors
    for fire in fires:
        for x, y in adj_cells(fire):
            # If the cell is open and we have not simulated it, simulate the fire spreading
            if valid_cell(x, y, maze) and (x, y) not in visited:
                if random.random() <= 1 - math.pow(1 - q, count_fires(maze, (x, y))):
                    # If fire spreads, update the new maze and new fires
                    new_maze[x][y] = 2
                    new_fires.append((x, y))
                visited.add((x, y))
        # If the cell we were checking is now surrounded by four fires,
        # it can no longer spread so we remove it from our list
        if count_fires(new_maze, fire) == 4:
            new_fires.remove(fire)

    return new_maze, new_fires


def dfs(maze, s, g):
    """
    Given a maze, performs DFS search algorithm starting from s
    and checks if cell g is reachable
    :param maze: The particular maze to check
    :param s: Tuple of coordinates, the starting coordinate
    :param g: Tuple of coordinates, the goal coordinate
    :return: Boolean on if G is reachable from S
    """

    for c in s + g:
        if not 0 <= c < len(maze):
            return

    temp = deepcopy(maze)
    temp[s[0]][s[1]] = 0
    temp[g[0]][g[1]] = 0

    # Generate stack structure for fringe, set for visited
    fringe = [s]
    visited = set()

    # While still cells to check, loop
    while fringe:
        current = fringe.pop()
        visited.add(current)
        # If cell is goal, then return True
        if current == g:
            return True
        # Check neighboring cells. If not visited and valid, add to fringe
        for x, y in adj_cells(current):
            if (x, y) not in visited and valid_cell(x, y, temp):
                fringe.append((x, y))

    # Goal cell not found and fringe is empty, return False
    return False


def bfs(maze, s, g, distances=False):
    """
    Given a maze, performs BFS search algorithm starting from s
    and checks if cell g is reachable. Return num visited cells.
    :param maze: The particular maze to check
    :param s: Tuple of coordinates, the starting coordinate
    :param g: Tuple of coordinates, the goal coordinate
    :param distances: If True, return map of shortest distances to s
    :return: Integer of the number of visited cells
    """

    for c in s + g:
        if not 0 <= c < len(maze):
            return

    # Use a deque to act as a queue for BFS
    fringe, visited = deque(), set()
    fringe.append(s)
    visited.add(s)
    num_visited = 0
    map = {}
    distance = 0

    num_in_layer = 1
    # While cells are in fringe
    while fringe:
        # Remove in FIFO fashion
        v = fringe.popleft()
        num_in_layer -= 1
        map[v] = distance
        num_visited += 1
        # If we found goal cell, break out of loop
        if not distances and v == g:
            break
        # For each neighbor, add to fringe if not visited and valid
        for x, y in adj_cells(v):
            if (x, y) not in visited and valid_cell(x, y, maze) and not (
                    distances and (x, y) == (len(maze) - 1, len(maze) - 1)):
                visited.add((x, y))
                fringe.append((x, y))
        if num_in_layer == 0:
            num_in_layer = len(fringe)
            distance += 1
            if distance > len(maze) * 2:
                break

    if distances:
        map.pop(s, None)  # we don't care about start
        return map
    # Return total number of cells BFS visited
    return num_visited


def a_star(maze, s, g, h_map, path=False):
    """
    Given a maze, performs A* search algorithm starting from s
    and checks if cell g is reachable. Return num visited cells.
    :param maze: The particular maze to check
    :param s: Tuple of coordinates, the starting coordinate
    :param g: Tuple of coordinates, the goal coordinate
    :param h_map: Euclidean distances from goal
    :return: Integer of the number of visited cells
    """

    for c in s + g:
        if not 0 <= c < len(maze):
            return
    if not dfs(maze, s, g):
        return None

    temp = deepcopy(maze)
    temp[s[0]][s[1]] = 0
    temp[g[0]][g[1]] = 0

    # Have all cells be infinite distance away except for start cell
    parent = [[None for _ in maze] for _ in maze]
    dist = [[(float("inf")) for _ in maze] for _ in maze]

    dist[s[0]][s[1]] = 0
    num_visited = 0
    visited = {s}
    # Use a heap data structure for the fringe
    fringe = [(0, s)]

    # While cells are in fringe
    while fringe:
        # Take cell closest out of the heap
        _, v = heapq.heappop(fringe)
        num_visited += 1
        # Break out if cell is goal cell
        if v == g:
            break
        dist_v = dist[v[0]][v[1]]
        # Check neighbors of the cell
        for x, y in adj_cells(v):
            if (x, y) not in visited and valid_cell(x, y, temp):
                visited.add(v)
                # If distance to neighbor cell through this cell is shorter, update distance in heap
                if dist_v + 1 < dist[x][y]:
                    dist[x][y] = dist_v + 1
                    parent[x][y] = (v[0], v[1])
                    heapq.heappush(fringe, (dist_v + 1 + h_map[(x, y)], (x, y)))

    if path:
        shortest_path = deque()
        shortest_path.append(g)
        prev = parent[g[0]][g[1]]
        if prev is None:
            return None
        while prev != s:
            shortest_path.appendleft(prev)
            prev = parent[prev[0]][prev[1]]
        return shortest_path
    else:
        # Returns number of cells visited in A*
        return num_visited


def scorch(maze, s, g, h_map, fires, q):
    '''
    simulated circumvention of risk convergence hotspots: simulates spread of fire several times and calculates
    chance of each cell being on fire when agent gets there. We use the word convergence since we want to know the
    probability that the agent and fire converge on that cell at the same time; if this chance is high, we want to
    block off this cell in a temp maze so the agent avoids it. Then keep blocking off as many dangerous cells as
    possible until the point where blocking another dangerous cell would block all paths to goal. Finally, run a* on
    this temp maze where the most dangerous cells are blocked.
    :param maze: 2d list representing maze this algo will run on
    :param s: tuple for agent location
    :param g: tuple for goal (dim-1, dim-1)
    :param h_map: map of euclidean distances from goal, for a* call
    :param fires: set of fire coordinates so we can simulate their spread
    :param q: flammability rate for fire spread that we will simulate
    :return: shortest path to goal that avoids dangerous cells
    '''
    for c in s + g:
        if not 0 <= c < len(maze):
            return None
    if not dfs(maze, s, g):
        return None

    # for each cell, find num steps it would take for agent to reach that cell (we use bfs for this)
    check_heap = []
    map = bfs(maze, s, g, distances=True)
    for (x, y) in map.keys():
        if s != (x, y) and g != (x, y):
            distance = map.get((x, y))
            heapq.heappush(check_heap, (distance, (x, y)))

    # simulate fire spread and calculate chance of each cell being on fire when agent arrives there
    block_heap = []
    if check_heap:
        num_trials = 10
        ignition_counts = {}
        for item in check_heap:
            cell = item[1]
            ignition_counts[cell] = 0.0
        saved_heap = check_heap.copy()
        for trial in range(num_trials):
            temp = deepcopy(maze)
            temp_fires = fires.copy()
            fire_step = 0
            check_heap = saved_heap.copy()
            while check_heap:
                temp, temp_fires = tick_maze(temp, temp_fires, q)
                fire_step += 1
                while check_heap and check_heap[0][0] == fire_step:
                    popped = heapq.heappop(check_heap)
                    x, y = popped[1]
                    # if cell is on fire or goal is on fire, count as failure
                    if temp[x][y] == 2 or temp[len(maze) - 1][len(maze) - 1] == 2:
                        ignition_counts[(x, y)] += 1.0
        for (x, y) in ignition_counts.keys():
            count = ignition_counts.get((x, y))
            prob = count / num_trials
            heapq.heappush(block_heap, (-1 * prob, (x, y)))  # -1 because we want max heap functionality

    '''
    deb = deepcopy(maze)
    for tuple in block_heap:
        prob = tuple[0] * -1
        x, y = tuple[1]
        deb[x][y] = prob
    deb[s[0]][s[1]] = 'A'
    for row in deb:
        print(row)
    print()
    '''

    # in temp maze, block as many dangerous cells as we can and then run a*
    temp = deepcopy(maze)
    while block_heap:
        popped = heapq.heappop(block_heap)
        x, y = popped[1]
        prob = popped[0] * -1
        if prob != 0.0:
            temp[x][y] = 1
            if not dfs(temp, s, g):
                temp[x][y] = 0
    return a_star(temp, s, g, h_map, path=True)
