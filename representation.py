#!/om/user/janner/anaconda2/envs/pytorch/bin/python

import os, argparse, numpy as np, torch, pdb
import pipeline, models, data, utils, visualization


parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='logs/trial/')  
parser.add_argument('--max_train', type=int, default=5000)
parser.add_argument('--max_test', type=int, default=500)

parser.add_argument('--model', type=str, default='full', choices=['full', 'no-gradient', 'cnn-lstm', 'uvfa-text'])
parser.add_argument('--annotations', type=str, default='human', choices=['synthetic', 'human'])
parser.add_argument('--mode', type=str, default='local', choices=['local', 'global'])

parser.add_argument('--map_dim', type=int, default=10)
parser.add_argument('--state_embed', type=int, default=1)
parser.add_argument('--obj_embed', type=int, default=7)

parser.add_argument('--lstm_inp', type=int, default=15)
parser.add_argument('--lstm_hid', type=int, default=30)
parser.add_argument('--lstm_layers', type=int, default=1)
parser.add_argument('--attention_kernel', type=int, default=3)
parser.add_argument('--attention_out_dim', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=int, default=0.001)
parser.add_argument('--iters', type=int, default=200)
args = parser.parse_args()


#################################
############## Data #############
#################################

train_data, test_data = data.load(args.mode, args.annotations, args.max_train, args.max_test)
layout_vocab_size, object_vocab_size, text_vocab_size, text_vocab = data.get_statistics(train_data, test_data)

print '\n<Main> Converting to tensors'
train_layouts, train_objects, train_rewards, train_terminal, \
        train_instructions, train_indices, train_values, train_goals = data.to_tensor(train_data, text_vocab)

test_layouts, test_objects, test_rewards, test_terminal, \
    test_instructions, test_indices, test_values, test_goals = data.to_tensor(test_data, text_vocab)

print '<Main> Training: (', train_layouts.size(), 'x', train_objects.size(), 'x', train_indices.size(), ') -->', train_values.size()
print '<Main> test     : (', test_layouts.size(), 'x', test_objects.size(), 'x', test_indices.size(), ') -->', test_values.size()


#################################
############ Training ###########
#################################

print '\n<Main> Initializing model: {}'.format(args.model)
model = models.init(args, layout_vocab_size, object_vocab_size, text_vocab_size)

train_inputs = (train_layouts, train_objects, train_indices)
test_inputs = (test_layouts, test_objects, test_indices)

print '<Main> Training model'
trainer = pipeline.Trainer(model, args.lr, args.batch_size)
trainer.train(train_inputs, train_values, test_inputs, test_values, iters=args.iters)


#################################
######## Save predictions #######
#################################

## make logging directories
pickle_path = os.path.join(args.save_path, 'pickle')
utils.mkdir(args.save_path)
utils.mkdir(pickle_path)

print '\n<Main> Saving model to {}'.format(args.save_path)
## save model
torch.save(model, os.path.join(args.save_path, 'model.pth'))

print '<Main> Saving predictions to {}'.format(pickle_path)
## save inputs, outputs, and MDP info (rewards and terminal maps)
pipeline.save_predictions(model, test_inputs, test_values, test_rewards, test_terminal, text_vocab, pickle_path, prefix='test_')


#################################
######### Visualization #########
#################################

vis_path = os.path.join(args.save_path, 'visualization')
utils.mkdir(vis_path)

print '<Main> Saving visualizations to {}'.format(vis_path)
## save images with predicted and target value maps
visualization.vis_predictions(model, test_inputs, test_values, test_instructions, vis_path, prefix='test_')








