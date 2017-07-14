import torch, torch.nn as nn

class ValuesFactorized(nn.Module):

    def __init__(self, state_model, goal_model):
        super(ValuesFactorized, self).__init__()
        
        self.state_model = state_model
        self.goal_model = goal_model

    # def forward_large(self, inputs, batch_size = 32):
        

    def forward(self, inp):
        state, objects, instructions = inp
        # print 'in Values: ', state.size(), objects.size(), instructions.size()
        state_embedding = self.state_model.forward(state)
        goal_embedding = self.goal_model.forward( (objects, instructions) )
        
        values = state_embedding * goal_embedding
        values = values.sum(1)
        # ## num_states x rank
        # num_states = state_embedding.size(0)
        # ## num_goals x rank
        # num_goals = goal_embedding.size(0)

        # ## num_states x num_goals x rank
        # state_rep = state_embedding.unsqueeze(1).repeat(1,num_goals,1)
        # goal_rep = goal_embedding.repeat(num_states,1,1)

        # values = state_rep * goal_rep
        # values = values.sum(2).squeeze()
        # values = values.transpose(0,1)

        return values





