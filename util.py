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
valid_cell = lambda x, y, maze, fire_point=False, val=0: 0 <= x < len(maze) and 0 <= y < len(maze) and (
    maze[x][y] == val if not fire_point else (maze[x][y] == val or maze[x][y] == 2))


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
        print('dim should be positive integer')
        return
    # Validate p
    if not 0 <= p <= 1:
        print('p should be float where 0 < p < 1')
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
    Simulates the fire spreading one step in some maze
    :param maze: The maze to simulate fire in
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
            if valid_cell(dx, dy, maze, 2):
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


def move_analysis(maze, q, current):
    """
    Calculates probabilities of agent's neighboring cells setting on fire
    :param maze: The maze to simulate fire in
    :param q: The flammability rate of the fire
    :param current: Agent's current location
    :return: Set of tuples containing possible moves and probability of each move resulting in immolation
    """

    def count_fires(maze, cell):
        """
        Helper function to count fires surrounding a cell
        :param maze: Maze to count from
        :param cell: Pair of coordinates of interest
        :return: Number of fires surrounding cell
        """
        num_fires = 0
        for x, y in adj_cells(cell):
            if valid_cell(x, y, maze, val=2):
                num_fires += 1
        return num_fires

    moves = set()
    for x, y in adj_cells(current):
        if valid_cell(x, y, maze):
            moves.add(((x, y), 1 - math.pow(1 - q, count_fires(maze, (x, y)))))
    return moves


def dfs(maze, s, g, fire_point=False):
    """
    Given a maze, performs DFS search algorithm starting from s
    and checks if cell g is reachable
    :param maze: The particular maze to check
    :param s: Tuple of coordinates, the starting coordinate
    :param g: Tuple of coordinates, the goal coordinate
    :param fire_point: If True, s or g is on fire and we should find shortest path anyways
    :return: Boolean on if G is reachable from S
    """

    for c in s + g:
        if not 0 <= c < len(maze):
            print("Coordinates out of bound")
            return

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
            if (x, y) not in visited and valid_cell(x, y, maze, fire_point):
                fringe.append((x, y))

    # Goal cell not found and fringe is empty, return False
    return False


def bfs(maze, s, g):
    """
    Given a maze, performs BFS search algorithm starting from s
    and checks if cell g is reachable. Return num visited cells.
    :param maze: The particular maze to check
    :param s: Tuple of coordinates, the starting coordinate
    :param g: Tuple of coordinates, the goal coordinate
    :return: Integer of the number of visited cells
    """

    for c in s + g:
        if not 0 <= c < len(maze):
            print("Coordinates out of bound")
            return

    # Use a deque to act as a queue for BFS
    fringe, visited = deque(), set()
    fringe.append(s)
    visited.add(s)
    num_visited = 0

    # While cells are in fringe
    while fringe:
        # Remove in FIFO fashion
        v = fringe.popleft()
        num_visited += 1
        # If we found goal cell, break out of loop
        if v == g:
            break
        # For each neighbor, add to fringe if not visited and valid
        for x, y in adj_cells(v):
            if (x, y) not in visited and valid_cell(x, y, maze):
                visited.add((x, y))
                fringe.append((x, y))

    # Return total number of cells BFS visited
    return num_visited


def a_star(maze, s, g, h_map, path=False):
    """
    Given a maze, performs A* search algorithm starting from s
    and checks if cell g is reachable. Return num visited cells.
    :param maze: The particular maze to check
    :param s: Tuple of coordinates, the starting coordinate
    :param g: Tuple of coordinates, the goal coordinate
    :return: Integer of the number of visited cells
    """

    for c in s + g:
        if not 0 <= c < len(maze):
            print("Coordinates out of bound")
            return

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
        # Check neighbors of the cell
        for x, y in adj_cells(v):
            if (x, y) not in visited and valid_cell(x, y, maze):
                visited.add(v)
                # If distance to neighbor cell through this cell is shorter, update distance in heap
                dist_v = dist[v[0]][v[1]]
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


def acronym(maze, s, g, h_map, fires, q):
    """
    Calculates probability of cells neighboring agent to set on fire, then picks safest cell
    as next move. If there is a tie, use a* to see which path is shorter.
    :param maze: The particular maze to check
    :param s: Tuple of coordinates, the starting coordinate
    :param g: Tuple of coordinates, the goal coordinate
    :param h_map: Heuristic
    :param fires: A set containing coordinates of every fire
    :param q: The flammability rate of the fire
    :return: Next best move or None if no moves left
    """

    for c in s + g:
        if not 0 <= c < len(maze):
            print('Coordinates out of bound')
            return
    if not dfs(maze, s, g):
        print('No path to goal anymore')
        return None
    moves = move_analysis(maze, q, s)  # each element in moves looks like ((x, y), prob of fire next tick)
    min_prob = 1.0
    for move in moves:
        prob = move[1]
        if prob < min_prob:
            min_prob = prob
    safest_moves = set()
    for move in moves:
        prob = move[1]
        # cell must have lowest prob of setting on fire out of agent's options
        if prob == min_prob:
            # check if the move can get to goal without going back to current cell
            temp_maze = deepcopy(maze)
            temp_maze[s[0]][s[1]] = 1
            if dfs(temp_maze, move[0], g):
                safest_moves.add(move[0])
    if len(safest_moves) == 1:
        return safest_moves.pop()
    max_distance = math.sqrt(math.pow(len(maze), 2) * 2)
    best_move = None
    for move in safest_moves:
        distance = h_map[move]
        if distance < max_distance:
            best_move = move
            max_distance = distance
    return best_move
