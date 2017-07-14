import torch, torch.nn as nn

class CompositorModel(nn.Module):

    def __init__(self, state_model, goal_model):
        super(CompositorModel, self).__init__()
        
        self.state_model = state_model
        self.goal_model = goal_model


    def forward(self, state, objects, instructions):
        state_embedding = self.state_model.forward(state)
        goal_embedding = self.goal_model.forward( (objects, instructions) )
        
        ## num_states x rank
        num_states = state_embedding.size(0)
        ## num_goals x rank
        num_goals = goal_embedding.size(0)

        ## num_states x num_goals x rank
        state_rep = state_embedding.unsqueeze(1).repeat(1,num_goals,1)
        goal_rep = goal_embedding.repeat(num_states,1,1)

        values = state_rep * goal_rep
        values = values.sum(2).squeeze()
        values = values.transpose(0,1)

        return values





