import sys, os, math, pickle, numpy as np, torch, pdb
from tqdm import tqdm
import environment.library as library


HUMAN_TEST_MAPS = {
    'local': [18,20,21,22,23,24,25,26,27,49,53,20,30,40,50,35,45,46,47],
    'global': [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,60,61,62,63,64,65,66,67,68,69,110,111,112,113,114,115,116,117,118,119]
}


def load(mode, data, num_train, num_test):
    turk_path = os.path.join('data', mode)
    states, _ = initialize_states(turk_path)

    if data == 'human':
        annotation_path = os.path.join('data', mode + '.p')
        test_maps = HUMAN_TEST_MAPS[mode]

        print '\n<Data> Loading {} train environments with human annotations'.format(mode)
        train = build_turk_observations_unified(turk_path, annotation_path, num_train, states, test_maps, test = False)
        
        print '\n<Data> Loading {} test environments with human annotations'.format(mode)
        test = build_turk_observations_unified(turk_path, annotation_path, num_test, states, test_maps, test = True)
         
    
    elif data == 'synthetic':
        raise RuntimeError('synthetic data is not cleaned up yet -- check back in a few days!')
        layouts, objects, rewards, terminal, instructions, values, goals = build_reinforce_observations_unified(turk_path, range(num_train), states)
        pdb.set_trace()

        val_layouts, val_objects, val_rewards, val_terminal, val_instructions, val_values, val_goals = decomposition.build_reinforce_observations_unified(args.test_path, range(args.num_test), states)

    return train, test

def synthetic_to_maps(map_dim):
    layouts, objects, rewards, terminal, instructions, values, goals = data
    layouts = layouts[0::map_dim**2]
    objects = objects[0::map_dim**2]
    instructions = instructions[0::map_dim**2]

    values = np.array(values).view(-1,map_dim,map_dim)

def get_statistics(train_data, val_data):
    ## unpack data
    train_layouts,  train_objects,  train_rewards,  train_terminal, train_instructions, train_values,   train_goals = train_data
    val_layouts,    val_objects,    val_rewards,    val_terminal,   val_instructions,   val_values,     val_goals   = val_data

    ## get word indices for all words in instructions
    full_instructions = train_instructions + val_instructions
    text_vocab = get_word_indices(full_instructions)

    ## get number of unique objects in layouts and objects
    layout_vocab_size = len( np.unique(train_layouts) )
    object_vocab_size = len( np.unique(train_objects) )
    ## add 1 for 0-padding of the text instructions
    text_vocab_size   = len(text_vocab) + 1

    return layout_vocab_size, object_vocab_size, text_vocab_size, text_vocab

def to_tensor(data, text_vocab):
    layouts, objects, rewards, terminal, instructions, values, goals = data
    # full_instructions = copy.deepcopy(train_instructions + val_instructions)

    # print '\nLoading {} synthetic environments'.format(args.num_synthetic)
    # if args.num_synthetic > 0:
    #     synth_layouts, synth_objects, synth_rewards, synth_terminal, synth_instructions, synth_values, synth_goals = decomposition.build_reinforce_observations_unified(args.synthetic_path, range(args.num_synthetic), states)
    #     full_instructions = full_instructions + synth_instructions

    ## get size of all "vocabularies"
    ## (number of objects, number of unique words, etc)
    # layout_vocab_size = len( np.unique(train_layouts) )
    # object_vocab_size = len( np.unique(train_objects) )
    # text_vocab = pipeline.word_indices(full_instructions)
    ## add 1 for 0-padding
    # text_vocab_size = len(text_vocab) + 1 
    indices = instructions_to_indices(instructions, text_vocab)
    # val_indices = pipeline.instructions_to_indices(val_instructions, text_vocab)

    # print '\nVocabulary'
    # print text_vocab
    # print 'Layout: {} | Objects: {} | Instructions: {} '.format(layout_vocab_size, object_vocab_size, text_vocab_size)

    # print '\nConverting to tensors'
    # num_loaded = turk_layouts.shape[0]
    # num_test = int(num_loaded * args.test_split)

    layouts = torch.Tensor(layouts).long().cuda()
    objects = torch.Tensor(objects).long().cuda()
    rewards = torch.Tensor(rewards).cuda()
    terminal = torch.Tensor(terminal).cuda()
    indices = torch.Tensor(indices).long().cuda()
    values = torch.Tensor(values).cuda()
    goals = torch.Tensor(goals).cuda()
    # instructions = train_instructions

    # if args.num_synthetic > 0:
    #     synth_indices = pipeline.instructions_to_indices(synth_instructions, text_vocab)

    #     ## only want the grass / puddles, not agent position
    #     synth_layouts = torch.Tensor(synth_layouts).long()[0:-1:args.map_dim**2,0].unsqueeze(1).cuda() #,1,:,:]
    #     synth_objects = torch.Tensor(synth_objects).long()[0:-1:args.map_dim**2].cuda()
    #     synth_indices = torch.Tensor(synth_indices).long()[0:-1:args.map_dim**2].cuda()
    #     synth_values = torch.Tensor(synth_values).cuda()
    #     synth_values = synth_values.view(-1,args.map_dim,args.map_dim)
    #     synth_goals = torch.Tensor(synth_goals[0:-1:args.map_dim**2]).cuda()

    #     # pdb.set_trace()
    #     layouts = torch.cat( (layouts, synth_layouts), 0)
    #     objects = torch.cat( (objects, synth_objects), 0)
    #     # print 'VAL: ', values.size(), synth_values.size(), goals.size(), synth_goals.size()
    #     values = torch.cat( (values, synth_values), 0)
    #     goals = torch.cat( (goals, synth_goals), 0)

    #     ## make synthetic instructions the same length as turk instructions
    #     turk_length = indices.size(1)
    #     num_synth = synth_indices.size(0)
    #     synth_length = synth_indices.size(1)
    #     synth_indices = torch.cat( (torch.zeros(num_synth, turk_length-synth_length).long().cuda(), synth_indices), 1)
       
    #     indices = torch.cat( (indices, synth_indices), 0)

    # val_layouts = torch.Tensor(val_layouts).long().cuda()
    # val_objects = torch.Tensor(val_objects).long().cuda()
    # val_rewards = val_rewards
    # val_terminal = val_terminal
    # val_indices = torch.Tensor(val_indices).long().cuda()
    # val_values = torch.Tensor(val_values).cuda()
    # val_goals = torch.Tensor(val_goals).cuda()

    return layouts, objects, rewards, terminal, instructions, indices, values, goals


########################################################################
######################## Instruction Processing ########################
########################################################################

'''
word indices start at 1, not 0
so that 0-padding is possible
'''
def get_word_indices(instructions):
    words = [word for phrase in instructions for word in phrase.split(' ')]
    unique = list(set(words))
    indices = { unique[i]: i+1 for i in range(len(unique)) }
    # num_unique = len(indices)
    # print indices
    return indices

'''
0-pads indices so that all sequences
are the same length
'''
def instructions_to_indices(instructions, ind_dict):
    ## split strings to list of words
    instructions = [instr.split(' ') for instr in instructions]
    num_instructions = len(instructions)
    max_instr_length = max([len(instr) for instr in instructions])
    indices_obs = np.zeros( (num_instructions, max_instr_length) )
    for count in range(num_instructions):
        indices = [ind_dict[word] for word in instructions[count]]
        # print indices
        indices_obs[count,-len(indices):] = indices
    # print 'num instr: ', num_instructions
    # print 'max length: ', max_instr_length
    # print indices_obs
    return indices_obs

########################################################################

'''
Returns dictionary mapping states of the form (i,j)
to an index representing the column in the value matrix.
Assumes there is a file '0.p' in the folder data_path/
'''
def initialize_states(data_path):
    path = os.path.join(data_path, '0.p')
    info = pickle.load( open(path, 'r') )
    shape = info['map'].shape

    states = [(i,j) for i in range(shape[0]) for j in range(shape[1])]
    num_states = len(states)
    states_dict = {states[i]: i for i in range(num_states)}
    return states, states_dict

'''
Returns a two-channel state observation 
where the first channel marks grass positions with 1's
and puddle positions with 0's
and the second channel marks the agent position with a 2
'''
def reconstruct_state(world, position):
    not_puddles = np.ma.masked_not_equal(world, library.objects['puddle']['index'])
    not_puddles = not_puddles.mask.astype(int)
    position_channel = np.zeros( (not_puddles.shape) )
    # indices 0 and 1 are used for puddle and grass, respectively
    # 2 is used for agent
    position_channel[position] = 2
    state = np.stack( (not_puddles, position_channel) )
    return state

'''
Returns single-channel goal observation with
objects denoted by unique indices beginning at 1.
0's denote background (no object)
'''
def reconstruct_goal(world):
    # pdb.set_trace()
    world = world.copy()
    ## indices for grass and puddle
    background_inds = [obj['index'] for (name, obj) in library.objects.iteritems() if obj['background']]
    ## background mask
    background = np.in1d(world, background_inds)
    background = background.reshape( (world.shape) )
    ## set backgronud to 0
    world[background] = 0
    ## subtract largest background ind
    ## so indices of objects begin at 1
    world[~background] -= max(background_inds)
    world = np.expand_dims(np.expand_dims(world, 0), 0)
    # pdb.set_trace()
    return world


'''
returns np array of size (len(worlds_list) * 400) x state_obs.shape
'''
def build_state_observations(data_path, worlds_list, states, states_dict, state_channels = 2):
    num_worlds = len(worlds_list)
    num_states = len(states_dict)
    state_size = int(math.sqrt(num_states))
    state_obs = np.zeros( (num_worlds * num_states, state_channels, state_size, state_size) )

    for ind, world_num in enumerate(worlds_list):
        path = os.path.join(data_path, str(world_num) + '.p')
        info = pickle.load( open(path, 'r') )
        world = info['map']

        count = ind * num_states
        for position in states:
            # print state_obs.shape, reconstruct_state(world, position).shape
            state_obs[count,:,:,:] = reconstruct_state(world, position)
            # print count, state
            count += 1

        # print count, state_obs.shape
    return state_obs

'''
returns layouts (w/ position), objects, instructions, and values 
'''
def build_value_observations_unified(data_path, worlds_list, states, state_channels = 2, goal_channels = 1, verbose = True):
    num_states = len(states)
    state_size = int(math.sqrt(num_states))

    state_obs = np.zeros( (0, state_channels, state_size, state_size) )
    object_obs = np.zeros( (0, goal_channels, state_size, state_size) )
    instruct_obs = []
    value_obs = []
    goals_obs = []

    if verbose:
        progress = tqdm(total = len(worlds_list))
    for world_ind, world_num in enumerate(worlds_list):
        if verbose:
            progress.update(1)
        path = os.path.join(data_path, str(world_num) + '.p')
        state_single, object_single, instruct_single, value_single, goals_single = \
            build_single(path, state_channels, goal_channels)

        state_obs = np.vstack( (state_obs, state_single) )
        object_obs = np.vstack( (object_obs, object_single) )
        instruct_obs.extend( instruct_single )
        value_obs.extend( value_single )
        goals_obs.extend( goals_single )
        # pdb.set_trace()

    return state_obs, object_obs, instruct_obs, value_obs, goals_obs

def build_reinforce_observations_unified(data_path, worlds_list, states, state_channels = 2, goal_channels = 1, verbose = True):
    num_states = len(states)
    state_size = int(math.sqrt(num_states))

    state_obs = np.zeros( (0, state_channels, state_size, state_size) )
    object_obs = np.zeros( (0, goal_channels, state_size, state_size) )
    reward_obs = np.zeros( (0, 1, state_size, state_size) )
    terminal_obs = np.zeros( (0, 1, state_size, state_size) )

    instruct_obs = []
    value_obs = []
    goals_obs = []

    if verbose:
        progress = tqdm(total = len(worlds_list))
    for world_ind, world_num in enumerate(worlds_list):
        if verbose:
            progress.update(1)
        path = os.path.join(data_path, str(world_num) + '.p')
        state_single, object_single, reward_single, terminal_single, \
            instruct_single, value_single, goals_single = \
            build_reinforce_single(path, state_channels, goal_channels)

        state_obs = np.vstack( (state_obs, state_single) )
        object_obs = np.vstack( (object_obs, object_single) )
        reward_obs = np.vstack( (reward_obs, reward_single) )
        terminal_obs = np.vstack( (terminal_obs, terminal_single) )

        instruct_obs.extend( instruct_single )
        value_obs.extend( value_single )
        goals_obs.extend( goals_single )
        # pdb.set_trace()

    return state_obs, object_obs, reward_obs, terminal_obs, instruct_obs, value_obs, goals_obs

def build_single(path, state_channels, goal_channels):
    info = pickle.load( open(path, 'r') )
    world = info['map']
    goals = info['goals']
    instructions = info['instructions']
    values = info['values']

    num_goals = len(goals)
    states = [(i,j) for i in range(world.shape[0]) for j in range(world.shape[1])]
    num_states = len(states)
    state_size = world.shape[0] ## assuming square states
    assert world.shape[0] == world.shape[1]

    state_obs = np.zeros( (num_goals*num_states, state_channels, state_size, state_size) )
    object_obs = np.zeros( (num_goals*num_states, goal_channels, state_size, state_size) )
    instruct_obs = []
    value_obs = []
    goal_obs = []

    objects = reconstruct_goal(world)
    
    count = 0
    for goal_ind, goal in enumerate(goals):
        val_map = values[goal_ind].flatten()
        instr = instructions[goal_ind]

        for pos_ind, position in enumerate(states):
            state = reconstruct_state(world, position)
            val = val_map[pos_ind]

            object_obs[count] = objects
            state_obs[count] = state
            instruct_obs.append( instr )
            value_obs.append( val )
            goal_obs.append( goal )

            count += 1
    assert count == num_goals * num_states
    return state_obs, object_obs, instruct_obs, value_obs, goal_obs

def build_reinforce_single(path, state_channels, goal_channels):
    info = pickle.load( open(path, 'r') )
    world = info['map']
    goals = info['goals']
    instructions = info['instructions']
    values = info['values']
    reward_maps = info['rewards']
    terminal_maps = info['terminal']

    num_goals = len(goals)
    states = [(i,j) for i in range(world.shape[0]) for j in range(world.shape[1])]
    num_states = len(states)
    state_size = world.shape[0] ## assuming square states
    assert world.shape[0] == world.shape[1]

    state_obs = np.zeros( (num_goals*num_states, state_channels, state_size, state_size) )
    object_obs = np.zeros( (num_goals*num_states, goal_channels, state_size, state_size) )
    reward_obs = np.zeros( (num_goals*num_states, 1, state_size, state_size) )
    terminal_obs = np.zeros( (num_goals*num_states, 1, state_size, state_size) )

    instruct_obs = []
    value_obs = []
    goal_obs = []

    objects = reconstruct_goal(world)
    
    count = 0
    for goal_ind, goal in enumerate(goals):
        val_map = values[goal_ind].flatten()
        instr = instructions[goal_ind]
        reward = reward_maps[goal_ind]
        terminal = terminal_maps[goal_ind]

        for pos_ind, position in enumerate(states):
            state = reconstruct_state(world, position)
            val = val_map[pos_ind]

            object_obs[count] = objects
            state_obs[count] = state
            reward_obs[count] = reward
            terminal_obs[count] = terminal

            instruct_obs.append( instr )
            value_obs.append( val )
            goal_obs.append( goal )

            count += 1
    assert count == num_goals * num_states
    return state_obs, object_obs, reward_obs, terminal_obs, instruct_obs, value_obs, goal_obs

def build_turk_observations_unified(data_path, turk_path, num_worlds, states, test_maps, test = False, state_channels = 1, goal_channels = 1, verbose = True):
    if type(test_maps) == str:
        test_maps = [int(i) for i in test_maps.split(',')]

    num_states = len(states)
    state_size = int(math.sqrt(num_states))

    state_obs = np.zeros( (0, state_channels, state_size, state_size) )
    object_obs = np.zeros( (0, goal_channels, state_size, state_size) )
    reward_obs = np.zeros( (0, 1, state_size, state_size) )
    terminal_obs = np.zeros( (0, 1, state_size, state_size) )
    value_obs = np.zeros( (0, state_size, state_size) )
    instruct_obs = []
    goals_obs = []

    annotations = pickle.load( open(turk_path, 'rb') )
    keys = annotations.keys()

    count = 0
    if verbose:
        progress = tqdm(total = num_worlds, leave=False)
    for key in keys:
        world_num, goal_num = key

        if (test and world_num not in test_maps) or (not test and world_num in test_maps): 
            continue
        if len(annotations[key]) < 5:
            continue
        _, instr, actual_goal, annotated_goal, accuracy = annotations[key]
        
        full_path = os.path.join(data_path, str(world_num) + '.p')
        if accuracy == 'wrong' or actual_goal != annotated_goal:
            continue
        else:
            progress.update(1)
            state_single, object_single, reward_single, terminal_single, value_single = \
                build_turk_single(full_path, actual_goal, state_channels, goal_channels)

            state_obs = np.vstack( (state_obs, state_single) )
            object_obs = np.vstack( (object_obs, object_single) )
            reward_obs = np.vstack( (reward_obs, reward_single) )
            terminal_obs = np.vstack( (terminal_obs, terminal_single) )
            value_obs = np.vstack( (value_obs, value_single) )
            instruct_obs.append( instr )
            goals_obs.append( actual_goal )
            count += 1

        if count == num_worlds:
            break

    # if count < num_worlds - 1:
        # print 'Only found {} annotations, breaking'.format(count)
    sys.stdout.flush()
    print '<Data> Found {} annotations'.format(count)
    return state_obs, object_obs, reward_obs, terminal_obs, instruct_obs, value_obs, goals_obs


def build_turk_single(path, goal, state_channels, goal_channels):
    info = pickle.load( open(path, 'r') )
    world = info['map']
    goals = info['goals']
    instructions = info['instructions']
    value_maps = info['values']
    reward_maps = info['rewards']
    terminal_maps = info['terminal']

    ## find goal
    goal_ind = goals.index(goal)

    reward = reward_maps[goal_ind][np.newaxis,np.newaxis]
    terminal = terminal_maps[goal_ind][np.newaxis,np.newaxis]
    values = value_maps[goal_ind][np.newaxis,:,:]

    pos = (0, 0)
    ## take position off of state
    state = reconstruct_state(world, pos)[0][np.newaxis,np.newaxis,:,:]
    objects = reconstruct_goal(world)

    # pdb.set_trace()

    return state, objects, reward, terminal, values

    # num_goals = len(goals)
    # states = [(i,j) for i in range(world.shape[0]) for j in range(world.shape[1])]
    # num_states = len(states)
    # state_size = world.shape[0] ## assuming square states
    # assert world.shape[0] == world.shape[1]

    # state_obs = np.zeros( (num_states, state_channels, state_size, state_size) )
    # object_obs = np.zeros( (num_states, goal_channels, state_size, state_size) )
    # reward_obs = np.zeros( (num_states, 1, state_size, state_size) )
    # terminal_obs = np.zeros( (num_states, 1, state_size, state_size) )

    # instruct_obs = []
    # value_obs = []
    # goal_obs = []

    # objects = reconstruct_goal(world)
    
    # count = 0
    # for goal_ind, goal in enumerate(goals):
    #     val_map = values[goal_ind].flatten()
    #     instr = instructions[goal_ind]
    #     reward = reward_maps[goal_ind]
    #     terminal = terminal_maps[goal_ind]

    #     for pos_ind, position in enumerate(states):
    #         state = reconstruct_state(world, position)
    #         val = val_map[pos_ind]

    #         object_obs[count] = objects
    #         state_obs[count] = state
    #         reward_obs[count] = reward
    #         terminal_obs[count] = terminal

    #         instruct_obs.append( instr )
    #         value_obs.append( val )
    #         goal_obs.append( goal )

    #         count += 1
    # assert count == num_goals * num_states
    # return state_obs, object_obs, reward_obs, terminal_obs, instruct_obs, value_obs, goal_obs

def build_mdp(data_path, worlds_list, MDP):
    mdp_list = []
    for world_num in worlds_list:
        path = os.path.join(data_path, str(world_num) + '.p')
        info = pickle.load( open(path) )
        world = info['map']
        rewards = info['rewards']
        terminal = info['terminal']
        goals = info['goals']

        # configurations = len(goals)
        for ind, goal in enumerate(goals):
            reward_map = rewards[ind]
            terminal_map = terminal[ind]
            mdp = MDP(world, reward_map, terminal_map)
            mdp_list.append(mdp)
    return mdp_list


'''
returns 
1. np array of size (len(worlds_list) * 400) x goal_obs.shape
2. list of instructions for each goal
3. target embedding for each goal
'''
def build_goal_observations(data_path, worlds_list, states_dict, V = None, goal_channels = 1):
    num_worlds = len(worlds_list)
    num_states = len(states_dict)
    state_size = int(math.sqrt(num_states))

    goal_obs = np.zeros( (0, goal_channels, state_size, state_size) )
    instruct_obs = []
    goal_columns = []
    ## check if goal embedding matrix V is provided
    if type(V) is np.ndarray:
        targets = np.zeros( (0, V.shape[1]) )
    else:
        targets = None
    values = np.zeros( (0, num_states) )

    for world_num in worlds_list:
        path = os.path.join(data_path, str(world_num) + '.p')
        info = pickle.load( open(path, 'r') )
        
        world = info['map']
        goals = info['goals']
        vals = info['values']
        instruct = info['instructions']
        configurations = len(goals)

        configurations = len(goals)
        for ind, goal in enumerate(goals):
            # pdb.set_trace()
            obs = reconstruct_goal(world)
            val_map = vals[ind].flatten()

            goal_obs = np.vstack( (goal_obs, obs) )
            instruct_obs.append( instruct[ind] )
            values = np.vstack( (values, val_map) )

            ## states_dict maps goal positions to column of value matrix
            goal_col = states_dict[goal]
            goal_columns.append(goal_col)
            if type(V) is np.ndarray:
                ## V is num_goals x rank
                targ = V[goal_col, :]
                targets = np.vstack( (targets, targ) )
            # pdb.set_trace()

    return goal_obs, instruct_obs, targets, values, goal_columns

def build_test_values(data_path, worlds_list, states):
    test_set = {}
    for world_num in worlds_list:
        layouts, objects, instructions, values = build_value_observations_unified(data_path, [world_num], states, verbose=False)
        test_set[world_num] = (layouts, objects, instructions, values)
    return test_set

'''
test_set[world_num] = (state_obs, goal_obs, instruct_obs, values)
where state_obs is state at every position, 
goal_obs is the same object observation repeated len(instruct_obs) times
instruct_obs is a list of instructions for the map in state_obs
and values is a np array of len(instruct_obs) x num_states

values[i,:] is the value of _all_ state_obs at goal_obs[i] and instruct[i]
'''
def build_test_set(data_path, worlds_list, states, states_dict):
    num_states = len(states_dict)
    state_size = int(math.sqrt(num_states))

    test_set = {}

    for world_num in tqdm(worlds_list):
        # print 'world num: ', world_num
        state_obs = build_state_observations(data_path, [world_num], states, states_dict)
        goal_obs, instruct_obs, _, values, _ = build_goal_observations(data_path, [world_num], states_dict)

        # print 'test set:', state_obs.shape, goal_obs.shape, len(instruct_obs), values.shape
        test_set[world_num] = (state_obs, goal_obs, instruct_obs, values)

    return test_set

'''
test set is returned by build_test_set
index_fn is a function that replaces instructions with indices
index_mapping is a dictionary from words --> indices
'''
def test_set_indices(test_set, index_fn, index_mapping):
    new_test = {}
    for key, (state_obs, goal_obs, instruct_obs, values) in test_set.iteritems():
        instruct_inds = index_fn(instruct_obs, index_mapping)
        # print len(instruct_obs), instruct_inds.shape
        new_test[key] = (state_obs, goal_obs, instruct_obs, instruct_inds, values)
    return new_test


def build_simulation_set(data_path, worlds_list, states, states_dict, MDP):
    num_states = len(states_dict)
    state_size = int(math.sqrt(num_states))

    test_set = {}

    for world_num in tqdm(worlds_list):
        # print 'world num: ', world_num
        state_obs = build_state_observations(data_path, [world_num], states, states_dict)
        goal_obs, instruct_obs, _, values, _ = build_goal_observations(data_path, [world_num], states_dict)
        mdps = build_mdp(data_path, [world_num], MDP)

        # print 'test set:', state_obs.shape, goal_obs.shape, len(instruct_obs), values.shape
        test_set[world_num] = (state_obs, goal_obs, instruct_obs, values, mdps)

    return test_set

'''
test set is returned by build_test_set
index_fn is a function that replaces instructions with indices
index_mapping is a dictionary from words --> indices
'''
def simulation_set_indices(test_set, index_fn, index_mapping):
    new_test = {}
    for key, (state_obs, goal_obs, instruct_obs, values, mdps) in test_set.iteritems():
        instruct_inds = index_fn(instruct_obs, index_mapping)
        # print len(instruct_obs), instruct_inds.shape
        new_test[key] = (state_obs, goal_obs, instruct_obs, instruct_inds, values, mdps)
    return new_test



