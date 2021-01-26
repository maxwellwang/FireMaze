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
2: 
3: 
4: 
5: 
6: 
7: 
8: 
'''


# maze: 2d list
# returns: nothing
def print_maze(maze):
    for row in maze:
        print(row)


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
            maze[i].append(1 if (i != 0 or j != 0) and (i != dim - 1 or j != dim - 1) and random.random() <= p else 0)
    return maze
