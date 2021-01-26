import random

'''
Representation:
- maze is 2d list of integers 0-2
- 0 means empty
- 1 means occupied by obstacle
- 2 means on fire
'''


# dim: length of maze edge
# p: obstacle density
# returns: maze 2d list
def generate_maze(dim, p):
    # create 2d list of length dim
    # top left and bottom right corners should be empty
    # use p to determine if each square is occupied
    maze = []
    for i in range(dim):
        maze.append([])
        for j in range(dim):
            if not ((i == 0 and j == 0) or (i == dim - 1 and j == dim - 1)) and random.random() <= p:
                maze[i].append(1)
            else:
                maze[i].append(0)
    return maze


# maze: 2d list
# returns: nothing
def print_maze(maze):
    for row in maze:
        print(row)
