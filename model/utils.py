import torch
import torch.nn as nn
from . import unetRNN6Layer_noBlock


model_map = {
    'unetRNN6Layer_noBlock': {
        'net':unetRNN6Layer_noBlock.SNN
    }
}


def getNetwork(model_type: str, simulation_params, data_params):
    model =  model_map[model_type]['net'](simulation_params, data_params)
    return model

