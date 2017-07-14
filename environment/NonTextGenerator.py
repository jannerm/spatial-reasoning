import math, random, pdb
import numpy as np
import library
from Generator import Generator

# print library.objects

class NonTextGenerator(Generator):

    def __init__(self, objects, shape = (20,20), goal_value = 3, num_steps = 50):
        self.objects = objects
        self.shape = shape
        self.goal_value = goal_value
        self.num_steps = num_steps

    def new(self):
        directions = {}
        world = self.__puddles(self.num_steps)

        # states = [(i,j) for i in range(world.shape[0]) \
        #                 for j in range(world.shape[1]) \
        #                 if world[i][j] != self.objects['puddle']['index'] ]

        ## we are using ALL of the locations as goals,
        ## even those in puddles
        states = [ (i,j) for i in range(world.shape[0]) \
                         for j in range(world.shape[1]) ]
        
        reward_maps = []
        terminal_maps = []
        goals = []
        for state in states:
            reward = np.zeros( world.shape )
            terminal = np.zeros( world.shape )
            reward[state] = self.goal_value
            terminal[state] = 1
            reward_maps.append(reward)
            terminal_maps.append(terminal)
            goals.append(state)

        info = {
            'map': world,
            'rewards': reward_maps,
            'terminal': terminal_maps,
            'goals': goals
        }

        # pdb.set_trace()

        return info

    def __puddles(self, iters, max_width=3, max_steps=10):
        (M,N) = self.shape
        turns = ['up', 'down', 'left', 'right']
        world = np.zeros( self.shape )
        world.fill( self.objects['puddle']['index'] )
        # position = np.floor(np.random.uniform(size=2)*self.shape[0]).astype(int)

        position = (np.random.uniform()*M, np.random.uniform()*N)
        position = map(int, position)

        for i in range(iters):
            direction = random.choice(turns)
            width = int(np.random.uniform(low=1, high=max_width))
            steps = int(np.random.uniform(low=1, high=max_steps))
            if direction == 'up':
                top = max(position[0] - steps, 0)
                bottom = position[0]
                left = max(position[1] - int(math.floor(width/2.)), 0)
                right = min(position[1] + int(math.ceil(width/2.)), N)
                position[0] = top
            elif direction == 'down':
                top = position[0] 
                bottom = min(position[0] + steps, M)
                left = max(position[1] - int(math.floor(width/2.)), 0)
                right = min(position[1] + int(math.ceil(width/2.)), N)
                position[0] = bottom
            elif direction == 'left':
                top = max(position[0] - int(math.floor(width/2.)), 0)
                bottom = min(position[0] + int(math.ceil(width/2.)), M)
                left = max(position[1] - steps, 0)
                right = position[1]
                position[1] = left
            elif direction == 'right':
                top = max(position[0] - int(math.floor(width/2.)), 0)
                bottom = min(position[0] + int(math.ceil(width/2.)), M)
                left = position[1] 
                right = min(position[1] + steps, N)
                position[1] = right
            # print top, bottom, left, right, self.objects['grass']['index']
            # print world.shape
            world[top:bottom+1, left:right+1] = self.objects['grass']['index']

        return world

    # def addRewards(self, world, positions, directions):
    #     reward_maps = []
    #     terminal_maps = []
    #     instruction_set = []
    #     goal_positions = []

    #     object_values = np.zeros( self.shape )
    #     ## add non-background values
    #     for name, obj in self.objects.iteritems():
    #         value = obj['value']
    #         if not obj['background']:
    #             pos = positions[name]
    #             object_values[pos] = value
    #         else:
    #             mask = np.ma.masked_equal(world, obj['index']).mask
    #             # print name, obj['index']
    #             # print mask
    #             # print value
    #             object_values[mask] += value
    #             # for st
    #             # value = obj['v']
    #     # print 'values: '
    #     # print object_values

    #     for name, obj in self.objects.iteritems():
    #         if not obj['background']:
    #         # ind = obj['index']
    #         # pos = positions[name]
    #             for (phrase, target_pos) in  directions[name]:
    #                 rewards = object_values.copy()
    #                 rewards[target_pos] += self.goal_value
    #                 terminal = np.zeros( self.shape )
    #                 terminal[target_pos] = 1

    #                 reward_maps.append(rewards)
    #                 terminal_maps.append(terminal)
    #                 instruction_set.append(phrase)
    #                 goal_positions.append(target_pos)

    #         # print name, pos, direct
    #     # print reward_maps
    #     # print instruction_set
    #     # print goal_positions
    #     return reward_maps, terminal_maps, instruction_set, goal_positions

    # # def randomPosition(self):
    # #     pos = tuple( (np.random.uniform(size=2) * self.dim).astype(int) )
    # #     return pos

    # def generateDirections(self, world, pos, name):
    #     directions = []
    #     for identifier, offset in self.directions.iteritems():
    #         phrase = 'reach cell ' + identifier + ' ' + name
    #         absolute_pos = tuple(map(sum, zip(pos, offset)))
    #         # print absolute_pos
    #         (i,j) = absolute_pos
    #         (M,N) = self.shape
    #         out_of_bounds = (i < 0 or i >= M) or (j < 0 or j >= N)

    #         # absolute_pos[0] < 0 or absolute_pos[0] >  [coord < 0 or coord >= self.dim for coord in absolute_pos]
    #         if not out_of_bounds:
    #             in_puddle = world[absolute_pos] == self.objects['puddle']['index']
    #             if not in_puddle:
    #                 directions.append( (phrase, absolute_pos) )
    #     # print directions
    #     return directions

    #     # for i in [-1, 0, 1]:
    #     #     for j in [-1, 0, 1]:
    #     #         if i == 0 and j == 0:
    #     #             pass
    #     #         else:


if __name__ == '__main__':
    gen = Generator(library.objects, library.directions)
    info = gen.new()

    print info['map']
    print info['instructions'], len(info['instructions'])
    print len(info['rewards']), len(info['terminal'])
    # print info['terminal']



