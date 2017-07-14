import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def visualize_values(mdp, values, policy, filename, title=None):
    states = mdp.states
    # print states
    plt.clf()
    m = max(states, key=lambda x: x[0])[0] + 1
    n = max(states, key=lambda x: x[1])[1] + 1
    data = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            state = (i,j)
            if type(values) == dict:
                data[i][j] = values[state]
            else:
                # print values[i][j]
                data[i][j] = values[i][j]
            action = policy[state]
            ## if using all_reachable actions, pick the best one
            if type(action) == tuple:
                action = action[0]
            if action != None:
                x, y, w, h = arrow(i, j, action)
                plt.arrow(x,y,w,h,head_length=0.4,head_width=0.4,fc='k',ec='k')
    heatmap = plt.pcolor(data, cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.gca().invert_yaxis()

    if title:
        plt.title(title)
    plt.savefig(filename + '.png')
    # print data

def arrow(i, j, action):
    ## up, down, left, right
    ## x, y, w, h
    arrows = {0: (.5,.95,0,-.4), 1: (.5,.05,0,.4), 2: (.95,.5,-.4,0), 3: (.05,.5,.4,0)}
    arrow = arrows[action]
    return j+arrow[0], i+arrow[1], arrow[2], arrow[3]

