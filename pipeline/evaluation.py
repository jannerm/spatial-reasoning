import os, math, pickle, torch, numpy as np, pdb, pipeline
from torch.autograd import Variable
import matplotlib; matplotlib.use('Agg')
from matplotlib import cm
from matplotlib import pyplot as plt
from tqdm import tqdm


'''
saves predictions of model, targets, and MDP info (rewards / terminal map) as pickle files
inputs is a tuple of (layouts, objects, instruction_indices)
assumes that save path already exists
'''
def save_predictions(model, inputs, targets, rewards, terminal, text_vocab, save_path, prefix=''):
    ## wrap tensors in Variables to pass to model
    input_vars = ( Variable(tensor.contiguous()) for tensor in inputs )
    predictions = model(input_vars)

    ## convert to numpy arrays for saving to disk
    predictions = predictions.data.cpu().numpy()
    targets = targets.cpu().numpy()

    ## save the predicted and target value maps
    ## as well as info about the MDP and instruction
    pickle.dump(predictions, open(os.path.join(save_path, prefix+'predictions.p'), 'wb') )
    pickle.dump(targets, open(os.path.join(save_path, prefix+'targets.p'), 'wb') )
    pickle.dump(rewards, open(os.path.join(save_path, prefix+'rewards.p'), 'wb') )
    pickle.dump(terminal, open(os.path.join(save_path, prefix+'terminal.p'), 'wb') )
    pickle.dump(text_vocab, open(os.path.join(save_path, prefix+'vocab.p'), 'wb') )


'''
test set is dict from
world number --> (state_obs, goal_obs, instruct_inds, values)
'''

def evaluate(model, test_set, savepath=None):
    progress = tqdm(total=len(test_set))
    count = 0
    for key, (state_obs, goal_obs, instruct_words, instruct_inds, targets) in test_set.iteritems():
        progress.update(1)
        
        state = Variable( torch.Tensor(state_obs).long().cuda() )
        objects = Variable( torch.Tensor(goal_obs).long().cuda() )
        instructions = Variable( torch.Tensor(instruct_inds).long().cuda() )
        targets = torch.Tensor(targets)
        # print state.size(), objects.size(), instructions.size(), targets.size()
        
        preds = model.forward( (state, objects, instructions) ).data.cpu()

        state_dim = 1
        for dim in state.size()[-2:]:
            state_dim *= dim

        if savepath:
            num_goals = preds.size(0) / state_dim
            for goal_num in range(num_goals):
                lower = goal_num * state_dim
                upper = (goal_num + 1) * state_dim
                fullpath = os.path.join(savepath, \
                            str(key) + '_' + str(goal_num) + '.png')
                pred = preds[lower:upper].numpy()
                targ = targets[lower:upper].numpy()
                instr = instruct_words[lower]

                pipeline.visualize_value_map(pred, targ, fullpath, title=instr)


def get_children(M, N):
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
def get_policy(values):
    values = values.squeeze()
    M, N = values.shape
    states = [(i,j) for i in range(M) for j in range(N)]
    children = get_children( M, N )
    policy = {}
    for state in states:
        reachable = children[state]
        selected = sorted(reachable, key = lambda x: values[x], reverse=True)
        policy[state] = selected[0]
    return policy

def simulate(model, sim_set):
    # progress = tqdm(total=len(test_set))
    steps_list = []
    count = 0
    for key in tqdm(range(len(sim_set))):
        (state_obs, goal_obs, instruct_words, instruct_inds, targets, mdps) = sim_set[key]
        # progress.update(1)
        # print torch.Tensor(state_obs).long().cuda()
        state = Variable( torch.Tensor(state_obs).long().cuda() )
        objects = Variable( torch.Tensor(goal_obs).long().cuda() )
        instructions = Variable( torch.Tensor(instruct_inds).long().cuda() )
        targets = torch.Tensor(targets)
        # print state.size(), objects.size(), instructions.size()
        
        preds = model.forward(state, objects, instructions).data.cpu().numpy()
        # print 'sim preds: ', preds.shape

        ## average over all goals
        num_goals = preds.shape[0]
        for ind in range(num_goals):
            # print ind
            mdp = mdps[ind]
            values = preds[ind,:]
            dim = int(math.sqrt(values.size))
            positions = [(i,j) for i in range(dim) for j in range(dim)]
            # print 'dim: ', dim
            values = preds[ind,:].reshape(dim, dim)
            policy = mdp.get_policy(values)

            # plt.clf()
            # plt.pcolor(policy)


            ## average over all start positions
            for start_pos in positions:
                steps = mdp.simulate(policy, start_pos)
                steps_list.append(steps)
                # pdb.set_trace()
                # print 'simulating: ', start_pos, steps
    avg_steps = np.mean(steps_list)
    # print 'avg steps: ', avg_steps, len(steps_list), len(sim_set), num_goals
    return avg_steps








