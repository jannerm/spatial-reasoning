import torch, pdb

def meta_rbf(size):
    batch = torch.zeros(size*size, size, size)
    count = 0
    for i in range(size):
        for j in range(size):
            batch[count,:,:] = rbf(size, (i,j)).clone()
            count += 1
    return batch

def rbf(size, position):
    x,y = position
    grid = torch.zeros(size, size)
    
    top_left = manhattan(size, row='increasing', col='increasing')
    bottom_left = manhattan(size, row='increasing', col='decreasing')
    top_right = manhattan(size, row='decreasing', col='increasing')
    bottom_right = manhattan(size, row='decreasing', col='decreasing')

    ## top left
    if x > 0 and y > 0:
        grid[:x+1, :y+1] = bottom_right[-x-1:, -y-1:]
    ## bottom left
    if x < size and y > 0:
        grid[x:, :y+1] = top_right[:size-x, -y-1:]
    ## top right
    if x > 0 and y < size:
        grid[:x+1, y:] = bottom_left[size-x-1:, :size-y]
    ## bottom right
    if x < size and y < size:
        grid[x:, y:] = top_left[:size-x, :size-y]

    return grid

def manhattan(size, row='increasing', col='increasing'):
    if row == 'increasing':
        rows = range_grid(0, size, 1, size)
    elif row == 'decreasing':
        rows = range_grid(size-1, -1, -1, size)
    else:
        raise RuntimeError('Unrecognized row in manhattan: ', row)

    if col == 'increasing':
        cols = range_grid(0, size, 1, size).t()
    elif col == 'decreasing':
        cols = range_grid(size-1, -1, -1, size).t()
    else:
        raise RuntimeError('Unrecognized col in manhattan: ', col)

    distance = rows + cols
    return distance

def range_grid(low, high, step, repeat):
    grid = torch.arange(low, high, step).repeat(repeat, 1)
    return grid



if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import pipeline
    
    map_dim = 10

    row = torch.arange(0,map_dim).unsqueeze(1).repeat(1,map_dim).numpy()
    col = torch.arange(0,map_dim).repeat(map_dim,1).numpy()

    pipeline.visualize_fig(row, 'figures/grad_row.png')
    pipeline.visualize_fig(col, 'figures/grad_col.png')

    # import scipy.misc
    # grid = rbf(10, (5,5))
    # print grid
    batch = meta_rbf(10).numpy()

    # pdb.set_trace()
    
    for b in range(batch.shape[0]):
        pipeline.visualize_fig(batch[b], 'figures/' + str(b) + '.png')
    print batch.size
    # print batch[-1]
    pdb.set_trace()










