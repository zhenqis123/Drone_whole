import torch
from torch import nn


class LSTMCell_new(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell_new, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for w in self.input_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.xavier_uniform_(w)
        for w in self.recurrent_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.orthogonal_(w)

    def forward(self, inputs, states):
        # 修改为对两个状态进行计算
        output_state_all, cell_state_all = states
        output_states = []
        cell_states = []
        for i in range(output_state_all.size(-1)):
            output_state = output_state_all[:, :, i]
            cell_state = cell_state_all[:, :, i]
            z = self.input_map(inputs) + self.recurrent_map(output_state)
            i, ig, fg, og = z.chunk(4, 1)

            input_activation = self.tanh(i)
            input_gate = self.sigmoid(ig)
            forget_gate = self.sigmoid(fg + 1.0)
            output_gate = self.sigmoid(og)

            new_cell = cell_state * forget_gate + input_activation * input_gate
            output_state = self.tanh(new_cell) * output_gate
            
            output_states.append(output_state)
            cell_states.append(new_cell)
        output_states = torch.stack(output_states, dim=-1)
        cell_states = torch.stack(cell_states, dim=-1)

        return output_states, cell_states
import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for w in self.input_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.xavier_uniform_(w)
        for w in self.recurrent_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.orthogonal_(w)

    def forward(self, inputs, states):
        output_state, cell_state = states
        z = self.input_map(inputs) + self.recurrent_map(output_state)
        i, ig, fg, og = z.chunk(4, 1)

        input_activation = self.tanh(i)
        input_gate = self.sigmoid(ig)
        forget_gate = self.sigmoid(fg + 1.0)
        output_gate = self.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = self.tanh(new_cell) * output_gate

        return output_state, new_cell