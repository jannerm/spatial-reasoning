#!/om/user/janner/anaconda2/envs/pytorch/bin/python

import sys, os, subprocess, argparse, numpy as np, pickle, pdb
from tqdm import tqdm
from matplotlib import cm
from vis_predictions import *
sys.path.append('../')
import environment, pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='../logs/example/pickle/') 
args = parser.parse_args()

predictions = pickle.load( open(os.path.join(args.save_path, 'predictions.p'), 'rb') ).squeeze()
targets = pickle.load( open(os.path.join(args.save_path, 'targets.p'), 'rb') ).squeeze()
rewards = pickle.load( open(os.path.join(args.save_path, 'rewards.p'), 'rb') ).squeeze()
terminal = pickle.load( open(os.path.join(args.save_path, 'terminal.p'), 'rb') ).squeeze()

vis_path = os.path.join(args.save_path, 'vis')
if not os.path.exists(vis_path):
    subprocess.call(['mkdir', vis_path])

num_worlds = targets.shape[0]

for ind in tqdm(range(num_worlds)):
    pred = predictions[ind]
    targ = targets[ind]

    vmax = max(pred.max(), targ.max())
    vmin = min(pred.min(), targ.min())

    pred_path = os.path.join(vis_path, str(ind) + '_pred.png')
    targ_path = os.path.join(vis_path, str(ind) + '_targ.png')

    vis_fig(pred, pred_path, vmax=vmax, vmin=vmin)
    vis_fig(targ, targ_path, vmax=vmax, vmin=vmin)


