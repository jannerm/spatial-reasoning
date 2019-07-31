import numpy as np, pdb

class ScoreIteration:

    def __init__(self, mdp, values):
        self.refresh(mdp, values)

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

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

    def refresh(self, mdp, values):
        self.mdp = mdp
        self.states = mdp.getStates()
        self.actions = mdp.getActions()
        self.transition = mdp.transition
        self.reward = mdp.reward
        self.terminal = mdp.terminal
        self.values = values
        self.scores = np.zeros( self.mdp.reward_map.shape )
        self.discount = 0.9


    def iterate(self):
        old_scores = self.scores.copy()
        old_scores.fill(-float('inf'))

        count = 0
        while np.sum(np.abs(self.scores - old_scores)) > 0.01:
            old_scores = self.scores.copy()
            for state in self.states:
                max_val = self.reward(state)
                term = self.terminal(state)
                
                if not term:
                    neighbors = [self.transition(state, action) for action in self.actions]
                    values = [self.values[s] for s in neighbors]
                    transition_probs = self._softmax(values)

                    neighbor_contributions = [transition_probs[ind] * self.scores[neighbors[ind]] \
                        for ind in range(len(transition_probs))]

                    max_val = self.reward(state) + self.discount * sum(neighbor_contributions)

                self.scores[state] = max_val
            count += 1

        avg_score = np.mean(self.scores)
        return avg_score, self.scores
