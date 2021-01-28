import random
from collections import deque
import heapq
from copy import deepcopy
import math
'''
Representation:
- maze is 2d list of integers 0-2
- 0 means empty
- 1 means occupied by obstacle
- 2 means on fire
'''

'''
Problems:
1: Maxwell - DONE
2: Maxwell
3: 
4: 
5: 
6: 
7: 
8: 
'''


# maze: 2d list
# return: nothing
def print_maze(maze):
    for row in maze:
        print(row)


# dim: length of maze edge
# p: obstacle density
# return: maze 2d list
def generate_maze(dim, p):
    # check if dim and p are valid
    # create 2d list of length dim
    # top left and bottom right corners should be empty
    # use p to determine if each cell is occupied
    if dim <= 0 or not isinstance(dim, int):
        print('dim should be positive integer')
        return
    if not 0 <= p <= 1:
        print('p should be float where 0 < p < 1')
        return
    maze = [[1 if random.random() < p else 0 for _ in range(dim)] for _ in range(dim)]
    maze[0][0], maze[-1][-1] = 0, 0
    return maze

def start_fire(maze):
    dim = len(maze)
    cell = random.randint(1, dim * dim - 1)
    while maze[cell//dim][cell % dim] != 0:
        cell = random.randint(1, dim * dim - 1)
    maze[cell // dim][cell % dim] = 2
    return ((cell//dim, cell % dim))

def adj_cells(x, y):
    dx, dy = [0, 0, -1, 1], [-1, 1, 0, 0]
    for i in range(4):
        yield (x + dx[i], y + dy[i])

def tick_maze(maze, fires, q):
    valid_cell = lambda x, y, dim: 0 <= x < dim and 0 <= y < dim
    def count_fires(maze, x, y):
        nonlocal valid_cell
        num_fires = 0
        for dx, dy in adj_cells(x, y):
            if valid_cell(dx, dy, len(maze)) and maze[dx][dy] == 2:
                num_fires += 1
        return num_fires

    new_fires = deepcopy(fires)
    new_maze = deepcopy(maze)
    visited = set()
    for fire in fires:
        for x, y in adj_cells(fire[0], fire[1]):
            if valid_cell(x, y, len(maze)) and (x, y) not in visited and maze[x][y] == 0:
                if random.random() <= 1 - math.pow(1-q, count_fires(maze, x, y)):
                    new_maze[x][y] = 2
                    new_fires.append((x, y))
                visited.add((x, y))
        if count_fires(new_maze, fire[0], fire[1]) == 4:
            new_fires.remove(fire)
    return (new_maze, new_fires)

# maze: 2d list
# current: current cell we want the neighbors of
# visited: visited cells that we should exclude
# return: neighbors of current cell that haven't been visited and are empty and in bounds
def get_neighbors(maze, current, visited):
    # check up/down and left/right cells
    # must be empty, in bounds, and unvisited to be valid neighbor
    neighbors = set()
    for i in [-1, 1]:
        x = current[0] + i
        y = current[1]
        if 0 <= x < len(maze) and (x, y) not in visited and maze[x][y] == 0:
            neighbors.add((x, y))
        x = current[0]
        y = current[1] + i
        if 0 <= y < len(maze) and (x, y) not in visited and maze[x][y] == 0:
            neighbors.add((x, y))
    return neighbors


# maze: 2d list
# s: start cell in tuple form
# g: goal cell in tuple form
# return: true if g reachable from s else false
def check_reachable(maze, s, g):
    # check if s and g are valid
    # use stack structure for fringe so it's DFS
    # keep track of visited
    dx, dy = [0, 0, -1, 1], [-1, 1, 0, 0]

    for c in s+g:
        if not 0 <= c < len(maze):
            print("Coordinates out of bound")
            return

    fringe = [s]
    visited = set()
    while fringe:
        current = fringe.pop()
        if current == g:
            return True
        # visited.add(current)
        for i in range(4):
            x, y = current[0] + dx[i], current[1] + dy[i]
            if (x, y) not in visited and 0 <= x < len(maze) and 0 <= y < len(maze) and maze[x][y] == 0:
                visited.add((x, y))
                fringe.append((x, y))
    return False


def bfs(maze, s, g):
    dx, dy = [0, 0, -1, 1], [-1, 1, 0, 0]

    fringe, visited = deque(), set()
    fringe.append(s)
    visited.add(s)
    num_visited = 0

    while fringe:
        v = fringe.popleft()
        num_visited += 1
        if v == g:
            break
        for i in range(4):
            x, y = v[0] + dx[i], v[1] + dy[i]
            if (x, y) not in visited and 0 <= x < len(maze) and 0 <= y < len(maze) and maze[x][y] == 0:
                visited.add((x, y))
                fringe.append((x, y))
    return num_visited

def a_star(maze, s, g):
    dx, dy = [0, 0, -1, 1], [-1, 1, 0, 0]
    h = lambda f, g : math.sqrt(math.pow(f[0]-g[0], 2) + math.pow(f[1]-g[1], 2))

    fringe = [(0, s)]
    dim = len(maze)
    dist = [[float("inf") for _ in range(dim)] for _ in range(dim)]
    dist[0][0] = 0
    visited = set()
    visited.add(s)
    num_visited = 0

    while fringe:
        v_dist, v = heapq.heappop(fringe)
        num_visited += 1
        if v == g:
            break
        for i in range(4):
            x, y = v[0] + dx[i], v[1] + dy[i]
            if (x, y) not in visited and 0 <= x < len(maze) and 0 <= y < len(maze) and maze[x][y] == 0:
                visited.add(v)
                new_dist = v_dist + 1 + h((x, y), g)
                if new_dist < dist[x][y]:
                    dist[x][y] = new_dist
                    heapq.heappush(fringe, (new_dist, (x, y)))

    return num_visited
