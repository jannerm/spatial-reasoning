import math, numpy as np, copy, pdb
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.autograd import Variable

class Agent:

    def __init__(self, network, target_network, lr=0.01, learn_start = 1000, batch_size = 32, map_dim = 10, gamma = 0.95, replay_size = 10000, instr_len = 7, layout_channels = 1, object_channels = 1):
        self.network = network
        self.target_network = target_network
        self._copy_net()
        self.learn_start = learn_start
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_size = replay_size
        self.instr_len = instr_len
        self.layout_channels = layout_channels
        self.object_channels = object_channels
        self._refresh_size(map_dim, map_dim)

        self.criterion = F.smooth_l1_loss
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=lr)

    def _refresh_size(self, M, N):
        self.M = M
        self.N = N
        self.states = self._get_states(M, N)
        self.children = self._get_children(M, N)
        self.replay_layouts = torch.Tensor(self.replay_size, self.layout_channels, M, N).long().cuda()
        self.replay_objects = torch.Tensor(self.replay_size, self.object_channels, M, N).long().cuda()
        self.replay_indices = torch.Tensor(self.replay_size, self.instr_len).long().cuda()
        self.replay_trajectories = [0 for i in range(self.replay_size)]
        self.replay_pointer = 0
        self.replay_filled = 0

    def _get_states(self, M, N):
        states = [(i,j) for i in range(M) for j in range(N)]
        return states

    def _get_children(self, M, N):
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
    for each state (i,j) in states, returns a list
    of neighboring states with approximated values 
    in descending order
    '''
    def _get_policy(self, values):
        values = values.squeeze()
        policy = {}
        for state in self.states:
            reachable = self.children[state]
            selected = sorted(reachable, key = lambda x: values[x], reverse = True)
            policy[state] = selected
        return policy

    '''
    copy parameters of network
    to target network
    '''
    def _copy_net(self):
        state = self.network.state_dict()
        state_clone = copy.deepcopy(state)
        self.target_network.load_state_dict(state_clone)


    def _get_targets(self, value_preds, trajectories):
        targets = torch.zeros(value_preds.size()).cuda()
        masks = torch.zeros(value_preds.size()).cuda()
        batch_size = value_preds.size(0)
        for b in range(batch_size):
            targ, mask = self._get_targets_single(value_preds[b], trajectories[b])
            targets[b] = targ
            masks[b] = mask
        return Variable(targets), Variable(masks)

        '''
    trajectory: [ (pos, val, rew, term), ... ]
    V(s) = R(s) + gamma * max_s' V(s')
    '''
    def _get_targets_single(self, value_map, trajectory):
        target = torch.zeros(self.M, self.N)
        mask = torch.zeros(self.M, self.N)

        value_map = value_map.data.squeeze()
        policy = self._get_policy(value_map)
        for (pos, _, rew, term) in trajectory:
            if term:
                max_v = 0
            ## find value of best neighbor
            else:
                best_reachable = policy[pos][0]
                max_v = value_map[best_reachable]
            target[pos] = rew + self.gamma * max_v
            mask[pos] = 1

        return target, mask


    def fill_replay(self, inputs, trajectories):
        layouts, objects, indices = inputs
        batch_size = len(trajectories)

        copy_len = min(batch_size, self.replay_size - self.replay_pointer)
        # print 'batch: ', batch_size, 'copy: ', copy_len
        start = self.replay_pointer
        end = self.replay_pointer + copy_len
        # print 'start: ', start, 'end: ', end
        self.replay_layouts[start:end] = layouts.data.clone()[:copy_len]
        self.replay_objects[start:end] = objects.data.clone()[:copy_len]
        self.replay_indices[start:end] = indices.data.clone()[:copy_len]
        self.replay_trajectories[start:end] = trajectories[:copy_len]
        # print self.replay_pointer, self.replay_filled
        self.replay_pointer = (self.replay_pointer + copy_len) % self.replay_size
        self.replay_filled = min(self.replay_filled + copy_len, self.replay_size)
        # print self.replay_pointer, self.replay_filled

    def sample_filled(self):
        layouts = self.replay_layouts[:self.replay_filled]
        objects = self.replay_objects[:self.replay_filled]
        indices = self.replay_indices[:self.replay_filled]
        inputs = (layouts, objects, indices)
        trajectories = self.replay_trajectories[:self.replay_filled]
        # inp, traj = self._get_batch( (layouts, objects, indices), trajectories, batch_size = size )
        return inputs, trajectories


    def simulate(self, values, rewards, terminals, max_steps = 75):
        # layouts, objects, indices = inputs
        # print worlds.size()
        # approx_values = self.network( inputs )
        # print 'APPROX: ', approx_values.size()
        batch_size = values.size(0)
        # print 'batch: ', batch_size
        trajectories = []
        scores = []
        for b in range(batch_size):
            # lay = layouts[b]
            # obj = objects[b]
            # ind = indices[b]
            reward_map = rewards[b]
            terminal_map = terminals[b]
            value_map = values[b]

            start_pos = self._random_pos()
            traj, rew = self._simulate_single(reward_map, terminal_map, value_map, start_pos, max_steps)
            # print 'got it!', type(traj), len(traj)
            trajectories.append(traj)
            scores.append(rew)
        return trajectories, np.mean(scores)

    def _simulate_single(self, reward_map, terminal_map, approx_values, start_pos, max_steps):
        # world = world.squeeze()
        reward_map = reward_map.data.squeeze()
        terminal_map = terminal_map.data.squeeze()
        approx_values = approx_values.data.squeeze()

        M, N = reward_map.size()
        # print 'M, N: ', M, N
        if M != self.M or N != self.N:
            raise RuntimeError( 'wrong size: {}x{}, expected: {}x{}'.format(M, N, self.M, self.N) )
            # self._refresh_size(M, N)
        policy = self._get_policy(approx_values)

        pos = start_pos
        visited = set([pos])
        trajectory = []

        total_reward = 0
        for step in range(max_steps):
            val = approx_values[pos]
            rew = reward_map[pos]
            term = terminal_map[pos]
            # print 'VAL: ', val, '\nREW: ', rew, '\nTERM: ', term, approx_values.size(), reward_map.size(), terminal_map.size()
            trajectory.append( (pos, val, rew, term) )

            total_reward += rew * (self.gamma ** step)
            if term:
                # print 'GOT REWARD: ', rew, step, rew * (self.gamma ** step)
                # print '\n\nDONE\n\n\n\n'
                break

            reachable = policy[pos]
            selected = 0
            while selected < len(reachable) and reachable[selected] in visited:
                # print '    visited ', selected, reachable[selected]
                selected += 1
            if selected == len(reachable):
                # print '\n\nVISITED ALL', pos, [n in visited for n in reachable], '\n\n\n\n'
                selected = 0
                # return trajectory
                break

            pos = reachable[selected]
            visited.add(pos)
            # print 'position: ', pos
        # print 'traj: ', len(trajectory), 'rew: ', total_reward 
        return trajectory, total_reward

    def _get_size(self, inputs):
        if type(inputs) == tuple:
            data_size = inputs[0].size(0)
        else:
            data_size = inputs.size(0)
        return data_size

    def _get_batch(self, inputs, targets, batch_size = None, volatile = False):
        data_size = self._get_size(inputs)

        if batch_size == None:
            batch_size = self.batch_size

        inds = torch.floor(torch.rand(batch_size) * data_size).long().cuda()
        # bug: floor(rand()) sometimes gives 1
        inds[inds >= data_size] = data_size - 1

        if type(inputs) == tuple:
            inp = tuple([Variable( i.index_select(0, inds).cuda(), volatile=volatile ) for i in inputs])
        else:
            inp = Variable( inputs.index_select(0, inds).cuda(), volatile=volatile )

        if type(targets) == list:
            targ = [targets[ind] for ind in inds]
        elif targets != None:
            targ = Variable( targets.index_select(0, inds).cuda(), volatile=volatile )
        else:
            targ = None

        return inp, targ

    def _epoch(self, inputs, trajectories, train_size):
        # print 'training with: ', [i.size() for i in inputs], len(trajectories)
        self.network.train()
        # data_size = self._get_size(inputs)
        num_batches = int( math.ceil(train_size / float(self.batch_size)) )

        err = 0
        for i in range(num_batches):
            inp, traj = self._get_batch(inputs, trajectories)
            self.optimizer.zero_grad()
            
            ## get predictions from the network
            ## and target network
            map_pred = self.network.forward(inp)
            map_targ = self.target_network.forward(inp)

            ## get targets for the value maps
            ## and mask for the states in the trajectories
            values_targ, map_mask = self._get_targets(map_targ, traj)
            values_pred = map_pred * map_mask

            loss = self.criterion(values_pred, values_targ)
            loss.backward()

            ## gradient clipping
            for param in self.network.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            err += loss.data[0]
        err = err / float(num_batches)
        return err

    def _random_pos(self):
        i = np.random.randint(self.M)
        j = np.random.randint(self.N)
        return (i, j)

    '''
    add num_environments environments 
    to the replay memory
    '''
    def act(self, mdp, num_environments):
        num_batches = int( math.ceil(float(num_environments) / self.batch_size) )
        score_sum = 0
        for i in range(num_batches):
            ## don't need gradients for simulation
            (lay, obj, ind, rew, term), _ = self._get_batch(mdp, None, volatile = True)
            
            ## get value estimations
            inputs = (lay, obj, ind)
            values = self.network(inputs)

            ## add inputs (layouts, objects, indices) 
            ## and trajectories to replay memory
            trajectories, score = self.simulate(values, rew, term)
            self.fill_replay(inputs, trajectories)
            score_sum += score
        avg_score = float(score_sum) / num_batches
        return avg_score


    def train(self, train_inputs, rewards, terminal, val_inputs, val_rewards, val_terminal, epochs = 1000):
        layouts, objects, indices = train_inputs
        val_layouts, val_objects, val_indices = val_inputs

        mdp_train = (layouts, objects, indices, rewards, terminal)
        mdp_eval = (val_layouts, val_objects, val_indices, val_rewards, val_terminal)

        ## populate replay memory
        self.act(mdp_train, self.learn_start)

        train_size = 500
        scores = []
        for i in range(epochs):
            # self.eval(val_inputs, val_rewards, val_terminal)

            ## add experiences
            score = self.act(mdp_train, train_size)
            ## get relevant part of replay memory
            inputs, trajectories = self.sample_filled()
            ## train on experiences
            err = self._epoch(inputs, trajectories, train_size)
            # print 'err: ', err
            print i, score
            scores.append(score)
            if i % 20 == 0:
                self._copy_net()

        return scores
            # self.target_network = copy.deepcopy(self.network)

        # pdb.set_trace()










if __name__ == '__main__':
    batch = 5
    dim = 10
    worlds = torch.randn(batch, 1, dim, dim)
    rewards = torch.ones(batch, 1, dim, dim) * 5
    terminals = torch.zeros(batch, 1, dim, dim)
    terminals[:,0,5,5] = 1
    network = nn.Conv2d(1,1,kernel_size=3,padding=1)
    approx = torch.randn(batch, 1, dim, dim)

    agent = Agent(network)
    agent.simulate(worlds, rewards, terminals)








