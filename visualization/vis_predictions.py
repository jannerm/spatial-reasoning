import os, math, torch, pdb
from tqdm import tqdm
from torch.autograd import Variable
import matplotlib; matplotlib.use('Agg')
from matplotlib import cm
from matplotlib import pyplot as plt

def vis_value_map(pred, targ, save_path, title='prediction', share=True):
    # print 'in vis: ', pred.shape, targ.shape
    dim = int(math.sqrt(pred.size))
    if share:
        vmin = min(pred.min(), targ.min())
        vmax = max(pred.max(), targ.max())
    else:
        vmin = None
        vmax = None

    plt.clf()
    fig, (ax0,ax1) = plt.subplots(1,2,sharey=True)
    heat0 = ax0.pcolor(pred.reshape(dim,dim), vmin=vmin, vmax=vmax, cmap=cm.jet)
    ax0.set_title(title, fontsize=5)
    if not share:
        fig.colorbar(heat0)
    heat1 = ax1.pcolor(targ.reshape(dim,dim), vmin=vmin, vmax=vmax, cmap=cm.jet)
    ax1.invert_yaxis()
    ax1.set_title('target')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(heat1, cax=cbar_ax)

    # print 'saving to: ', fullpath
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # print pred.shape, targ.shape

def vis_fig(data, save_path, title=None, vmax=None, vmin=None, cmap=cm.jet):
    # print 'in vis: ', pred.shape, targ.shape
    dim = int(math.sqrt(data.size))

    # if share:
    #     vmin = min(pred.min(), targ.min())
    #     vmax = max(pred.max(), targ.max())
    # else:
    #     vmin = None
    #     vmax = None

    plt.clf()
    # fig, (ax0,ax1) = plt.subplots(1,2,sharey=True)
    plt.pcolor(data.reshape(dim,dim), vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    # ax0.set_title(title, fontsize=5)
    # if not share:
        # fig.colorbar(heat0)
    # heat1 = ax1.pcolor(targ.reshape(dim,dim), vmin=vmin, vmax=vmax, cmap=cm.jet)
    fig = plt.gcf()
    ax = plt.gca()

    if title:
        ax.set_title(title)
    ax.invert_yaxis()

    fig.set_size_inches(4,4)

    # ax1.invert_yaxis()
    # ax1.set_title('target')

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(heat1, cax=cbar_ax)

    # print 'saving to: ', fullpath
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)

    # print pred.shape, targ.shape

def vis_predictions(model, inputs, targets, instructions, save_path, prefix=''):
    ## wrap tensors in Variables to pass to model
    input_vars = ( Variable(tensor.contiguous()) for tensor in inputs )
    predictions = model(input_vars)

    ## convert to numpy arrays for saving to disk
    predictions = predictions.data.cpu().numpy()
    targets = targets.cpu().numpy()

    num_inputs = inputs[0].size(0)
    for ind in tqdm(range(num_inputs)):
        pred = predictions[ind]
        targ = targets[ind]
        instr = instructions[ind]

        full_path = os.path.join(save_path, prefix + str(ind) + '.png')

        vis_value_map(pred, targ, full_path, title=instr, share=False)

