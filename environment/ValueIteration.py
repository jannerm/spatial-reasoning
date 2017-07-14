class ValueIteration:

    def __init__(self, mdp):
        self.refresh(mdp)

    def refresh(self, mdp):
        self.mdp = mdp
        self.states = mdp.getStates()
        self.actions = mdp.getActions()
        self.transition = mdp.transition
        self.reward = mdp.reward
        self.terminal = mdp.terminal
        self.values = {state: 0 for state in self.states}
        self.policy = {state: None for state in self.states}
        self.discount = 0.9


    def iterate(self):
        for k in range(0, 1000):
            for state in self.states:
                max_val = -float('inf')
                term = self.terminal(state)
                if term:
                    max_val = self.reward(state)
                else:
                    for action in self.actions:
                        new_state = self.transition(state, action)
                        new_val = self.reward(state) + self.discount * self.values[new_state]
                        if new_val > max_val:
                            max_val = new_val
                            self.policy[state] = action
                self.values[state] = max_val
        return self.values, self.policy