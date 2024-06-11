import numpy as np
import torch.nn as nn
from ncps.wirings import AutoNCP
from ncps.wirings import NCP
from ncps.torch import LTC
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import seaborn as sns
out_features = 2
in_features = 2

wiring = NCP(
    inter_neurons=18,  # Number of inter neurons
    command_neurons=12,  # Number of command neurons
    motor_neurons=4,  # Number of motor neurons
    sensory_fanout=6,  # How many outgoing synapses has each sensory neuron
    inter_fanout=4,  # How many outgoing synapses has each inter neuron
    recurrent_command_synapses=4,  # Now many recurrent synapses are in the
    # command neuron layer
    motor_fanin=6,# random seed to generate connections between nodes
    seed=22222
)

rnn = LTC(1    , wiring, batch_first=True)
rnn._wiring.build(1)
sns.set_style("white")
plt.figure(figsize=(10, 8))
legend_handles = wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()
plt.savefig('test.svg')