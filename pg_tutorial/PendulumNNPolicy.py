from parameterized_policy import ParameterizedGaussianPolicy

class PendulumNNPolicy:
    def __init__(self, state_dim, action_dim, action_range):
        self.policynn = ParameterizedGaussianPolicy(state_dim, action_dim, action_range)
        self.valuenn = ParameterizedGaussianPolicy(state_dim, action_dim, action_range)

    def get_action(self, state):
        return self.policynn.get_action(state)

    def get_value(self, state):
        return self.valuenn.get_action(state)

    def grad_value(self, state):
        return self.valuenn.sample_action(state)[1]

    def sample_action(self, state):
        return self.policynn.sample_action(state)

    def get_parameters_vector(self):
        return self.policynn.get_parameters_vector()
    def get_value_parameters_vector(self):
        return self.valuenn.get_parameters_vector()

    def set_parameters_vector(self, parameters_vector):
        self.policynn.set_parameters_vector(parameters_vector)
    def set_value_parameters_vector(self, value_parameters_vector):
        self.valuenn.set_parameters_vector(value_parameters_vector)
