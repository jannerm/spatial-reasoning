import numpy as np

class MDP:

    def __init__(self, world, rewards, terminal):
        self.world = world
        self.reward_map = rewards
        self.terminal_map = terminal
        self.shape = self.reward_map.shape

        self.M, self.N = self.shape
        self.states = [(i,j) for i in range(self.M) for j in range(self.N)]
        self.children = self.get_children( self.M, self.N )

        self.actions = [(-1,0),(1,0),(0,-1),(0,1)]
        self.states = [(i,j) for i in range(self.shape[0]) for j in range(self.shape[1])]

    def getActions(self):
        return [i for i in range(len(self.actions))]

    def getStates(self):
        return self.states

    def transition(self, position, action_ind, fullstate=False):
        action = self.actions[action_ind]
        # print 'transitioning: ', action, position
        candidate = tuple(map(sum, zip(position, action)))
        
        ## if new location is valid, 
        ## update the position
        if self.valid(candidate):
            position = candidate
        
        if fullstate:
            state = self.observe(position)
        else:
            state = position

        return state

    def valid(self, position):
        x, y = position[0], position[1]
        if x >= 0 and x < self.shape[0] and y >= 0 and y < self.shape[1]:
            return True
        else:
            return False

    def reward(self, position):
        rew = self.reward_map[position]
        return rew

    def terminal(self, position):
        term = self.terminal_map[position]
        return term

    def representValues(self, values):
        value_map = np.zeros( self.shape )
        for pos, val in values.iteritems():
            assert(value_map[pos] == 0)
            value_map[pos] = val
        return value_map

    # '''
    # start_pos is (i,j)
    # policy is dict from (i,j) --> (delta_i, delta_j)
    # '''
    # def simulate(self, policy, start_pos, num_steps = 100):
    #     pos = start_pos
    #     visited = set([pos])
    #     for step in range(num_steps):
    #         # rew = self.reward(pos)
    #         term = self.terminal(pos)
    #         if term:
    #             return 0
    #         reachable = policy[pos]
    #         selected = 0
    #         while selected < len(reachable) and reachable[selected] in visited:
    #             # print '    visited ', selected, reachable[selected]
    #             selected += 1
    #         if selected == len(reachable):
    #             selected = 0
    #         pos = policy[pos][selected]
    #         visited.add(pos)
    #         print 'position: ', pos
    #     # print pos, goal
    #     goal = np.argwhere( self.terminal_map ).flatten().tolist()
    #     manhattan_dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    #     return manhattan_dist

    '''
    start_pos is (i,j)
    policy is dict from (i,j) --> (delta_i, delta_j)
    '''
    def simulate(self, policy, start_pos, num_steps = 100):
        pos = start_pos
        visited = set([pos])
        for step in range(num_steps):
            # rew = self.reward(pos)
            term = self.terminal(pos)
            if term:
                return step
            reachable = policy[pos]
            selected = 0
            while selected < len(reachable) and reachable[selected] in visited:
                # print '    visited ', selected, reachable[selected]
                selected += 1
            if selected == len(reachable):
                selected = 0
            pos = policy[pos][selected]
            visited.add(pos)
            # print 'position: ', pos
        return step

    def get_children(self, M, N):
        children = {}
        for i in range(M):
            for j in range(N):
                pos = (i,j)
                children[pos] = []
                for di in range( max(i-1, 0), min(i+1, M-1)+1 ):
                    for dj in range( max(j-1, 0), min(j+1, N-1)+1 ):
                        child = (di, dj)
                        if pos != child and (i == di or j == dj):
                            children[pos].append( child )
        return children


    '''
    values is M x N map of predicted values
    '''
    def get_policy(self, values):
        policy = {}
        for state in self.states:
            reachable = self.children[state]
            selected = sorted(reachable, key = lambda x: values[x], reverse=True)
            policy[state] = selected
        return policy



if __name__ == '__main__':
    import pickle, pdb, numpy as np
    info = pickle.load( open('../data/train_10/0.p') )
    print info.keys()
    mdp = MDP(info['map'], info['rewards'][0], info['terminal'][0])
    print mdp.world
    print mdp.children

    values = np.random.randn(10,10)
    policy = mdp.get_policy(values)

    steps = mdp.simulate(policy, (0,0))
    print steps

    pdb.set_trace()


