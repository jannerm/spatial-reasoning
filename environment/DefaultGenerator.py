import math, random, pdb
import numpy as np
import library
from Generator import Generator

# print library.objects

class DefaultGenerator(Generator):

    def __init__(self, objects, directions, shape = (20,20), goal_value = 3, num_steps = 50):
        self.objects = objects
        self.directions = directions
        self.shape = shape
        self.goal_value = goal_value
        self.num_steps = num_steps

        self.default_world = np.array( 
                                        [   [1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1],
                                            [1,1,1,0,0,0,0,0,0,1,0,1,1,1,0,1,0,1,0,0],
                                            [1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,1,0,1,0,1],
                                            [1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,1,1],
                                            [0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,1,1,1,1],
                                            [0,0,0,1,1,0,1,0,0,0,0,0,1,0,1,1,1,1,1,1],
                                            [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1],
                                            [0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,1],
                                            [0,0,0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0],
                                            [0,0,0,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0],
                                            [0,1,1,1,1,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                                            [1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0],
                                            [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                            [1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                            [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                            [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                            [1,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                            [1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                            [1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]   ]
                                    )
        self.default_world = self.default_world * -1 + 1
        self.default_world = self.default_world[:10,:10]
        print 'world size: ', self.default_world.shape

    def new(self):
        directions = {}
        world = self.default_world.copy()
        # print world

        states = [(i,j) for i in range(world.shape[0]) \
                        for j in range(world.shape[1]) \
                        if world[i][j] != self.objects['puddle']['index'] ]

        used_indices = set( np.unique(world).tolist() )
        positions = {}
        for name, obj in self.objects.iteritems():
            # print obj['name']
            if not obj['background']:
                ind = obj['index']
                assert( ind not in used_indices )


                # if name == 'square':
                    # pos = (2,1)
                # else:
                pos = random.choice(states) #self.randomPosition()
                while pos in positions.values():
                    pos = random.choice(states)

                world[pos] = ind
                used_indices.add(ind)

                # print name
                positions[name] = pos
                object_specific_dir = self.generateDirections(world, pos, name)
                # print object_specific_dir
                directions[name] = object_specific_dir
                # print directions
                # pdb.set_trace()

        reward_maps, terminal_maps, instructions, goals = \
            self.addRewards(world, positions, directions)
        
        info = {
            'map': world,
            'rewards': reward_maps,
            'terminal': terminal_maps,
            'instructions': instructions,
            'goals': goals
        }
        return info
