#!/om/user/janner/anaconda2/envs/pytorch/bin/python

## norbf
# python run_eval.py --save_path rl_logs_analysis/qos_1250epochs_reinforce_norbfmodel_localmode_1566turk_0.001lr_0.95gamma_4batch_1rep/
# python run_eval.py --save_path rl_logs_analysis/qos_reinforce_norbfmodel_globalmode_1566turk_0.001lr_0.95gamma_4batch_1rep/

## uvfa-text
# python run_eval.py --save_path rl_logs_analysis//reinforce_uvfa-textmodel_localmode_1566turk_0.001lr_0.95gamma_32batch_3rep/
# python run_eval.py --save_path rl_logs_analysis//reinforce_uvfa-textmodel_globalmode_1566turk_0.001lr_0.95gamma_32batch_2rep/

## cnn-lstm
# run_eval.py --save_path rl_logs_analysis/reinforce_cnn-lstmmodel_localmode_1566turk_0.001lr_0.95gamma_16batch_2rep/
# python run_eval.py --save_path rl_logs_analysis/reinforce_cnn-lstmmodel_globalmode_1566turk_0.001lr_0.95gamma_32batch_3rep/

import sys, os, subprocess, argparse, numpy as np, pickle, pdb
sys.path.append('/om/user/janner/mit/urop/direction_decomposition/')
import environment, pipeline, reinforce

parser = argparse.ArgumentParser()
# parser.add_argument('--save_path', type=str, default='logs/trial_nobases_test') 
parser.add_argument('--save_path', type=str, default='curves/local_full_1500_wsynth-0-18,20,21,22,23,24,25,26,27,49,53,20,30,40,50,35,45_4/') 
# parser.add_argument('--metric_path', type=str, default='curves/local_full_1500_wsynth-0-18,20,21,22,23,24,25,26,27,49,53,20,30,40,50,35,45_4/') 
args = parser.parse_args()
args.metric_path = args.save_path

print args.save_path
predictions = pickle.load( open(os.path.join(args.save_path, 'test_predictions.p'), 'rb') ).squeeze()
targets = pickle.load( open(os.path.join(args.save_path, 'test_targets.p'), 'rb') ).squeeze()
rewards = pickle.load( open(os.path.join(args.save_path, 'test_rewards.p'), 'rb') ).squeeze()
terminal = pickle.load( open(os.path.join(args.save_path, 'test_terminal.p'), 'rb') ).squeeze()

rewards = rewards.cpu().numpy()
terminal = terminal.cpu().numpy()

def get_states(M, N):
    states = [(i,j) for i in range(M) for j in range(N)]
    return states

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
for each state (i,j) in states, returns a list
of neighboring states with approximated values 
in descending order
'''
def get_policy(values):
    values = values.squeeze()
    policy = {}
    for state in STATES:
        reachable = CHILDREN[state]
        selected = sorted(reachable, key = lambda x: values[x], reverse = True)
        policy[state] = selected
    return policy

def simulate_single(reward_map, terminal_map, approx_values, start_pos, max_steps = 75):
    # world = world.squeeze()
    # reward_map = reward_map.data.squeeze()
    # terminal_map = terminal_map.data.squeeze()
    # approx_values = approx_values

    M, N = reward_map.shape
    # print 'M, N: ', M, N
    if M != 10 or N != 10:
        raise RuntimeError( 'wrong size: {}x{}, expected: {}x{}'.format(M, N, self.M, self.N) )
        # self._refresh_size(M, N)
    policy = get_policy(approx_values)

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

        total_reward += rew * (GAMMA ** step)
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
    return total_reward

STATES = get_states(10,10)
CHILDREN = get_children(10,10)
GAMMA = 0.95

# pdb.set_trace()
quality_path = os.path.join(args.metric_path, 'quality')
if not os.path.exists( quality_path ):
    subprocess.call(['mkdir', quality_path])

num_worlds = targets.shape[0]
print 'Num worlds: {}'.format(num_worlds)

mse = np.sum(np.power(predictions - targets, 2)) / predictions.size
print 'MSE: {}'.format(mse)

cumulative_normed = 0
manhattan = 0
cumulative_per_score = 0
cumulative_score = 0
for ind in range(num_worlds):
    pred = predictions[ind]
    targ = targets[ind]

    pred_max = np.unravel_index(np.argmax(pred), pred.shape)
    targ_max = np.unravel_index(np.argmax(targ), targ.shape)
    man = abs(pred_max[0] - targ_max[0]) + abs(pred_max[1] - targ_max[1])

    # pdb.set_trace()

    unif = np.ones( pred.shape )
    rew = rewards[ind]
    term = terminal[ind]

    mdp = environment.MDP(None, rew, term)
    si = pipeline.ScoreIteration(mdp, pred)
    avg_pred, scores_pred = si.iterate()

    mdp = environment.MDP(None, rew, term)
    si = pipeline.ScoreIteration(mdp, targ)
    avg_targ, scores_targ = si.iterate()

    mdp = environment.MDP(None, rew, term)
    si = pipeline.ScoreIteration(mdp, unif)
    avg_unif, scores_unif = si.iterate()

    avg_per_score = np.divide(scores_pred-scores_unif, scores_targ-scores_unif)
    avg_per_score[avg_per_score != avg_per_score] = 1
    avg_per_score = np.mean(avg_per_score)
    # pdb.set_trace()

    start_pos = (np.random.randint(10), np.random.randint(10))
    score = 0 #simulate_single(rew, term, pred, start_pos)

    normed = (avg_pred - avg_unif) / (avg_targ - avg_unif)
    cumulative_normed += normed
    manhattan += man
    cumulative_per_score += avg_per_score
    cumulative_score += score

    sys.stdout.write( '{:4}:\t{:4}\t{:4}\t{:4}\t{:4}\t{:4}\t{:.4}\t{:.4}\t{:.3}\r'.format(ind, avg_pred, avg_targ, avg_unif, normed, 
        cumulative_normed / (ind + 1), cumulative_per_score / (ind + 1), float(manhattan) / (ind + 1), float(cumulative_score) / (ind + 1)))
    sys.stdout.flush()


    # fullpath = os.path.join(quality_path, str(ind) + '.png')
    # pipeline.visualize_value_map(scores_pred, scores_targ, fullpath)

    




    # pred_policy = pipeline.get_policy(pred)
    # targ_policy = pipeline.get_policy(targ)



    # corr = 0
    # for pos, targ in targ_policy.iteritems():
    #     pred = pred_policy[pos]
    #     if pred == targ:
    #         # print 'count: ', pred, targ
    #         corr += 1
    #     # else:
    #         # print 'not counting: ', pred, targ
    # print corr, ' / ', len(targ_policy)
    # cumulative_correct += corr

avg_normed = float(cumulative_normed) / num_worlds
avg_manhattan = float(manhattan) / num_worlds
avg_score = float(cumulative_score) / num_worlds
print 'Avg normed: {}'.format(avg_normed)
print 'Avg manhattan: {}'.format(avg_manhattan)
print 'Avg score: {}'.format(avg_score)

if args.metric_path != None:
    results = {'mse': mse, 'quality': avg_normed, 'manhattan': avg_manhattan}
    pickle.dump(results, open(os.path.join(args.metric_path, 'results.p'), 'wb'))










