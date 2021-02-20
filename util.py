import math, random, heapq
from collections import deque

'''
Representation:
- maze is 2d list of integers 0-2
- 0 means empty
- 1 means occupied by obstacle
- 2 means on fire
'''

"""
Helper lambda function to check cell is valid
:param x: x coordinate of cell
:param y: y coordinate of cell
:param maze: The 2d array to check the particular cell on
:param val: Value that a cell should be to be valid, default 0
:return: Boolean of validity of cell
"""
valid_cell = lambda x, y, maze, val = 0 : 0 <= x < len(maze) and 0 <= y < len(maze) and (val is None or maze[x][y] == val)


def adj_cells(pair):
    """
    Helper function to return cells adjacent to input
    :param pair: Coordinates of some cell
    :return: Generator which produces the four adjacent coordinates
    """
    dx, dy = [0, -1, 0, 1], [-1, 0, 1, 0]
    for i in range(4):
        yield (pair[0] + dx[i], pair[1] + dy[i])


def print_maze(maze, agent = None):
    """
    :param maze: Maze to be printed
    :param agent: cell that the agent is currently on, None by default
    :return: None
    """

    if agent:
        maze[agent[0]][agent[1]] = 3

    symbols = {
        0 : ".",
        1 : "X",
        2 : "F",
        3 : "A"
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
    while maze[cell//dim][cell % dim] != 0:
        cell = random.randint(1, dim * dim - 2)
    # Update cell with fire, then return
    maze[cell // dim][cell % dim] = 2
    return ((cell//dim, cell % dim))


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
                if random.random() <= 1 - math.pow(1-q, count_fires(maze, (x, y))):
                    # If fire spreads, update the new maze and new fires
                    new_maze[x][y] = 2
                    new_fires.append((x, y))
                visited.add((x, y))
        # If the cell we were checking is now surronded by four fires,
        # it can no longer spread so we remove it from our list
        if count_fires(new_maze, fire) == 4:
            new_fires.remove(fire)

    new_fires = list(set(new_fires))
    return (new_maze, new_fires)

def sim_maze(maze, fires, q):
    """
    Simulates the fire spreading one step in some maze
    :param maze: The maze to simulate fire in
    :param fires: A set containing coordinates of every fire
    :param q: The flammability rate of the fire
    :return: Tuple containing the new maze and new set of fires
    """
    def prob_fires(maze, fire):
        """
        Helper function to count fires surronding a cell
        :param maze: Maze to count from
        :param fire: Pair of coordinates of interest
        :return: Number of fires surronding cell
        """
        nonlocal q

        fires = []
        for dx, dy in adj_cells(fire):
            if valid_cell(dx, dy, maze, None):
                if maze[dx][dy] > 1:
                    fires.append(maze[dx][dy] - 1)

        # Count number of fires, take the average
        num_fires = len(fires)
        avg = sum(fires) / num_fires
        # Apply the approximation we have
        prob = avg * q * (0.3 + num_fires * 0.7)
        # Fix case if cell already on fire
        cur_prob = maze[fire[0]][fire[1]]
        if cur_prob != 0:
            cur_prob -= 1

        # Adjust probability cell is on fire
        prob = cur_prob + (1 - cur_prob) * prob

        # Round probability if needed
        if prob > 0.95:
            prob = 1
        if prob < 0.065:
            prob = 0
        return prob


    # Generate a copy of current fires and maze
    new_fires = fires[:]
    new_maze = [row[:] for row in maze]
    visited = set()
    to_remove = set()
    # For each fire, we check its neighbors
    for fire in fires:
        for x, y in adj_cells(fire):
            # If the cell is open and we have not simulated it, simulate the fire spreading
            if valid_cell(x, y, maze, None) and (x, y) not in visited:
                visited.add((x, y))
                if maze[x][y] != 1 and maze[x][y] != 2:
                    prob_fire = prob_fires(maze, (x, y))
                    # If fire spreads, update the new maze and new fires
                    if prob_fire > 0:
                        new_maze[x][y] = 1 + prob_fire
                        new_fires.append((x, y))
                        # Checks if fire can be removed from active fire list (if surronded by fires)
                        if prob_fire == 1:
                            num = 0
                            for w, z in adj_cells(fire):
                                if valid_cell(w, z, new_maze, None):
                                    if new_maze[w][z] == 1 or new_maze[w][z] == 2:
                                        num += 1
                            if num == 4:
                                to_remove.add(fire)
    new_fires = list(set(new_fires) - to_remove)
    return (new_maze, new_fires)


def dfs(maze, s, g):
    """
    Given a maze, performs DFS search algorithm starting from s
    and checks if cell g is reachable
    :param maze: The particular maze to check
    :param s: Tuple of coordinates, the starting coordinate
    :param g: Tuple of coordinates, the goal coordinate
    :return: Boolean on if G is reachable from S
    """

    for c in s+g:
        if not 0 <= c < len(maze):
            print("Coordinates out of bound")
            return

    # Generate stack structure for fringe, set for visited
    fringe = [s]
    visited = set()

    # While still cells to check, loop
    while fringe:
        current = fringe.pop()
        # If cell is goal, then return True
        if current == g:
            return True
        # Check neighboring cells. If not visited and valid, add to fringe
        for x, y in adj_cells(current):
            if (x, y) not in visited and valid_cell(x, y, maze):
                visited.add((x, y))
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

# SCORCH Algorithm
def a_star(maze, s, g, h_map, f = None, r = None, path = False):
    """
    Given a maze, performs A* search algorithm starting from s
    and checks if cell g is reachable. Return num visited cells.
    :param maze: The particular maze to check
    :param s: Tuple of coordinates, the starting coordinate
    :param g: Tuple of coordinates, the goal coordinate
    :param h_map: A mapping from each cell to some heuristic helper value
    :param f: An array of future mazes with expected fire probabilities
    :param r: The round offset, aka how "old" the future maze is (since we do not compute every round)
    :param path: Wether or not to return the path or just the number of nodes visited
    :return: Integer of the number of visited cells
    """

    # Have all cells be infinite distasnce away except for start cell
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
                    # f is for SCORCH, where we have future maze fire predictions
                    if f:
                        # Calculate number of steps to look into the future
                        taxi = int(math.fabs(s[0] - x) + math.fabs(s[1] - y))
                        # Push onto the heap the weight from the heuristic
                        heapq.heappush(fringe, (dist_v + 1 + f[min(round((taxi - 1)*1) + r, len(maze)*2-3)][x][y] * 50 + math.fabs(g[0] - x) + math.fabs(g[1] - y), (x, y)))
                    else:
                        # If not using SCORCH, just use the given heuristic instead
                        heapq.heappush(fringe, (dist_v + 1 + h_map[(x, y)], (x, y)))

    # If looking for the path, return it, if not return cells visited.
    if path:
        shortest_path = deque()
        shortest_path.append(g)
        prev = parent[g[0]][g[1]]
        # No parent assigned to goal, so it was not reached
        if prev == None:
            return None
        # There is a parent, keep following the chain to the start and return
        while prev != s:
            shortest_path.appendleft(prev)
            prev = parent[prev[0]][prev[1]]
        return shortest_path
    else:
        # Returns number of cells visited in A*
        return num_visited

