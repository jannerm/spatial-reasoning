#!/om/user/janner/anaconda2/envs/pytorch/bin/python

import sys, os, argparse, pickle, subprocess, pdb
from tqdm import tqdm
import environment, utils

parser = argparse.ArgumentParser()
parser.add_argument('--lower', type=int, default=0)
parser.add_argument('--num_worlds', type=int, default=10)
parser.add_argument('--vis_path', type=str, default='data/example_vis/')
parser.add_argument('--save_path', type=str, default='data/example_env/')
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--mode', type=str, default='local', choices=['local', 'global'])
parser.add_argument('--only_global', type=bool, default=False)
parser.add_argument('--sprite_dim', type=int, default=100)
parser.add_argument('--num_steps', type=int, default=10)
args = parser.parse_args()

print args, '\n'

utils.mkdir(args.vis_path)
utils.mkdir(args.save_path)


if args.mode == 'local':
    from environment.NonUniqueGenerator import NonUniqueGenerator
    gen = NonUniqueGenerator( environment.figure_library.objects, environment.figure_library.unique_instructions, shape=(args.dim, args.dim), num_steps=args.num_steps, only_global=args.only_global )
elif args.mode == 'global':
    from environment.GlobalGenerator import GlobalGenerator
    gen = GlobalGenerator( environment.figure_library.objects, environment.figure_library.unique_instructions, shape=(args.dim, args.dim), num_steps=args.num_steps, only_global=args.only_global )


for outer in range(args.lower, args.lower + args.num_worlds):
    info = gen.new()
    configurations = len(info['rewards'])

    print 'Generating map', outer, '(', configurations, 'configuations )'
    sys.stdout.flush()

    world = info['map']
    rewards = info['rewards']
    terminal = info['terminal']
    values = []

    sprite = environment.SpriteFigure(environment.figure_library.objects, environment.figure_library.background, dim=args.sprite_dim)
    sprite.makeGrid(world, args.vis_path + str(outer) + '_sprites')

    for inner in tqdm(range(configurations)):
        reward_map = rewards[inner]
        terminal_map = terminal[inner]

        mdp = environment.MDP(world, reward_map, terminal_map)
        vi = environment.ValueIteration(mdp)

        values_list, policy = vi.iterate()
        value_map = mdp.representValues(values_list)
        values.append(value_map)


    info['values'] = values
    filename = os.path.join( args.save_path, str(outer) + '.p' )
    pickle.dump( info, open( filename, 'wb' ) )


