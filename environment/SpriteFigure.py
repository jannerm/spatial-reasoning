import os, numpy as np, scipy.misc, pdb

class SpriteFigure:

    def __init__(self, objects, background, dim = 20):
        self.objects = objects
        self.dim = dim
        background = self.loadImg(background)
        self.sprites = {0: background}
        for name, obj in objects.iteritems():
            ind = obj['index']
            # print name
            sprite = self.loadImg( obj['sprite'] )
            if not obj['background']:
                # overlay = background.copy()
                # masked = np.ma.masked_greater( sprite[:,:,-1], 0 ).mask
                # overlay[masked] = sprite[:,:,:-1][masked]
                overlay = sprite
            else:
                overlay = sprite

            self.sprites[ind] = overlay


    def loadImg(self, path, dim = None):
        if dim == None:
            dim = self.dim
        path = os.path.join('environment', path)    
        img = scipy.misc.imread(path)
        img = scipy.misc.imresize(img, (dim, dim) )
        return img

    def makeGrid(self, world, filename, boundary_width = 4):
        shape = world.shape

        grass_ind = self.objects['grass']['index']
        puddle_ind = self.objects['puddle']['index']

        state = self.loadImg( self.objects['grass']['sprite'], dim = shape[0] * self.dim )
        puddle = self.loadImg( self.objects['puddle']['sprite'], dim = shape[0] * self.dim )

        for i in range(shape[0]):
            for j in range(shape[1]):

                row_low = i*self.dim
                row_high = (i+1)*self.dim
                col_low = j*self.dim
                col_high = (j+1)*self.dim

                ind = int(world[i,j])
                sprite = self.sprites[ind]

                if ind == grass_ind:
                    continue
                elif ind == puddle_ind:
                    state[row_low:row_high, col_low:col_high, :] = puddle[row_low:row_high, col_low:col_high, :]
                ## background
                else:
                    masked = np.ma.masked_greater( sprite[:,:,-1], 0 ).mask
                    state[row_low:row_high, col_low:col_high, :][masked] = sprite[:,:,:-1][masked]
                    # overlay[masked] = sprite[:,:,:-1][masked]
                # sprite = self.sprites[world[i,j].astype('int')]
                # state[i*self.dim:(i+1)*self.dim, j*self.dim:(j+1)*self.dim, :] = sprite

        for i in range(shape[0]):
            for j in range(shape[1]):

                row_low = i*self.dim
                row_high = (i+1)*self.dim
                col_low = j*self.dim
                col_high = (j+1)*self.dim

                ind = int(world[i,j])

                if i < shape[0] - 1:
                    below = int(world[i+1,j])
                    if (ind != puddle_ind and below == puddle_ind) or (ind == puddle_ind and below != puddle_ind):
                        # print 'BELOW: ', i, j
                        state[row_high-boundary_width:row_high+boundary_width, col_low:col_high, :] = 0.

                if j < shape[1] - 1:
                    right = int(world[i,j+1])
                    if (ind != puddle_ind and right == puddle_ind) or (ind == puddle_ind and right != puddle_ind):
                        # print 'BELOW: ', i, j
                        state[row_low:row_high, col_high-boundary_width:col_high+boundary_width, :] = 0.


        scipy.misc.imsave(filename + '.png', state)
        return state
