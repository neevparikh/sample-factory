import torch
from torch import nn

from algorithms.base.model_utils import create_encoder, create_core 
from algorithms.utils.action_distributions import sample_actions_log_probs, is_continuous_action_space
from utils.timing import Timing
from utils.utils import AttrDict


class _DQNBase(nn.Module):
    def __init__(self, action_space, cfg, timing):
        super().__init__()
        self.cfg = cfg
        self.action_space = action_space
        self.timing = timing
        self.encoders = []
        self.cores = []

    def model_to_device(self, device):
        self.to(device)
        for e in self.encoders:
            e.model_to_device(device)

    def device_and_type_for_input_tensor(self, input_tensor_name):
        return self.encoders[0].device_and_type_for_input_tensor(input_tensor_name)

    def initialize_weights(self, layer):
        # gain = nn.init.calculate_gain(self.cfg.nonlinearity)
        gain = 1.0

        if self.cfg.policy_initialization == 'orthogonal':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:  # TODO: test for LSTM
                nn.init.orthogonal_(layer.weight_ih, gain=gain)
                nn.init.orthogonal_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
            else:
                pass
        elif self.cfg.policy_initialization == 'xavier_uniform':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:
                nn.init.xavier_uniform_(layer.weight_ih, gain=gain)
                nn.init.xavier_uniform_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
            else:
                pass



class _SimpleDQN(_DQNBase):
    def __init__(self, make_encoder, make_core, action_space, cfg, timing):
        super().__init__(action_space, cfg, timing)

        # in case of shared weights we're using only a single encoder and a single core
        self.encoder = make_encoder()
        self.encoders = [self.encoder]

        self.core = make_core(self.encoder)
        self.cores = [self.core]

        core_out_size = self.core.get_core_out_size()
        self.q_tail = nn.Linear(core_out_size, self.action_space.n)

        self.apply(self.initialize_weights)
        self.train()  # eval() for inference?

    def forward_head(self, obs_dict):
        x = self.encoder(obs_dict)
        return x

    def forward_core(self, head_output):
        return self.core(head_output)

    def forward_tail(self, core_output):
        q_values = self.q_tail(core_output)

        result = AttrDict(dict(
            q_values=q_values,
        ))

        return result

    def forward(self, obs_dict):
        x = self.forward_head(obs_dict)
        x = self.forward_core(x)
        result = self.forward_tail(x)
        return result


class _DQN:
    def __init__(self, main, target):
        self.main = main
        self.target = target


def create_dqn(cfg, obs_space, action_space, timing=None):
    if timing is None:
        timing = Timing()

    def make_encoder():
        return create_encoder(cfg, obs_space, timing)

    def make_core(encoder):
        return create_core(cfg, encoder.get_encoder_out_size())

    main = _SimpleDQN(make_encoder, make_core, action_space, cfg, timing)
    target = _SimpleDQN(make_encoder, make_core, action_space, cfg, timing)
    return _DQN(main, target)
