import random

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
    if p <= 0 or p >= 1 or not isinstance(p, float):
        print('p should be float where 0 < p < 1')
        return
    maze = []
    for i in range(dim):
        maze.append([])
        for j in range(dim):
            maze[i].append(1 if (i != 0 or j != 0) and (i != dim - 1 or j != dim - 1) and random.random() <= p else 0)
    return maze


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
    if s[0] < 0 or s[0] >= len(maze) or s[1] < 0 or s[1] >= len(maze):
        print('s coordinates are out of bounds')
        return
    if g[0] < 0 or g[0] >= len(maze) or g[1] < 0 or g[1] >= len(maze):
        print('g coordinates are out of bounds')
        return
    fringe = [s]
    visited = set()
    while fringe:
        current = fringe.pop()
        if current == g:
            return True
        visited.add(current)
        fringe.extend(get_neighbors(maze, current, visited))
    return False
