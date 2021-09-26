import os
import torch
from model import getNetwork
from .gpu import moveToGPUDevice


class RBase:
    def __init__(self, datafile, log_config, general_config):
        self.datafile = datafile

        self.log_config = log_config
        self.genconfigs = general_config


        self.device = 0
        self.dtype = None 

    def _loadNetFromCheckpoint(self, ckpt_file, modeltype):
        assert ckpt_file
        checkpoint = torch.load(ckpt_file)
        self.net = getNetwork(modeltype, self.genconfigs['simulation'], self.genconfigs['data'])

        self.net.load_state_dict(checkpoint['state_dict'])
        moveToGPUDevice(self.net, self.device, self.dtype)
        self.log_config.copyModelFile(self.net)
