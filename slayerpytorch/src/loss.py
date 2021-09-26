import torch
import math
import numpy as np
import torch
import torch.nn as nn
from .slayer import spikeLayer
import cv2
from PIL import Image
import os

class spikeLoss(torch.nn.Module):   
    '''
    This class defines different spike based loss modules that can be used to optimize the SNN.

    NOTE: By default, this class uses the spike kernels from ``slayer.spikeLayer`` (``snn.layer``).
    In some cases, you may want to explicitly use different spike kernels, for e.g. ``slayerLoihi.spikeLayer`` (``snn.loihi``).
    In that scenario, you can explicitly pass the class name: ``slayerClass=snn.loihi`` 

    Usage:

    >>> error = spikeLoss.spikeLoss(networkDescriptor)
    >>> error = spikeLoss.spikeLoss(errorDescriptor, neuronDesc, simulationDesc)
    >>> error = spikeLoss.spikeLoss(netParams, slayerClass=slayerLoihi.spikeLayer)
    '''
    def __init__(self, errorDescriptor, neuronDesc, simulationDesc, slayerClass=spikeLayer):
        super(spikeLoss, self).__init__()
        self.neuron = neuronDesc
        self.simulation = simulationDesc
        self.errorDescriptor = errorDescriptor
        self.slayer = slayerClass(self.neuron, self.simulation)
        
    def __init__(self, networkDescriptor, slayerClass=spikeLayer):
        super(spikeLoss, self).__init__()
        self.neuron = networkDescriptor['neuron']
        self.simulation = networkDescriptor['simulation']
        self.slayer = slayerClass(self.neuron, self.simulation)

    def crop_like(input, target):
        
        if input.size()[2:] == target.size()[2:]:
            return input
        else:
            return input[:, :target.size(1), :target.size(2), :target.size(3)]     

    def spikeTime(self, spikeOut, spikeDesired):
        '''
        Calculates spike loss based on spike time.
        The loss is similar to van Rossum distance between output and desired spike train.

        .. math::

            E = \int_0^T \\left( \\varepsilon * (output -desired) \\right)(t)^2\\ \\text{d}t 

        Arguments:
            * ``spikeOut`` (``torch.tensor``): spike tensor
            * ``spikeDesired`` (``torch.tensor``): desired spike tensor

        Usage:

        >>> loss = error.spikeTime(spikeOut, spikeDes)
        '''
        # Tested with autograd, it works
        # assert self.errorDescriptor['type'] == 'SpikeTime', "Error type is not SpikeTime"
        # error = self.psp(spikeOut - spikeDesired) 
        error = self.slayer.psp(spikeOut - spikeDesired) 
        return 1/2 * torch.sum(error**2) * self.simulation['Ts']

    def MembraneSpikeTime(self, mask, spikeOut, spikeDesired, membraneOut, theta=0.22):
        XORmask = torch.logical_xor(mask, spikeOut)

        error = torch.mul(XORmask, membraneOut - theta * spikeDesired) 
        return 1/2 * torch.sum(error**2) * self.simulation['Ts']

    def numSpikes(self, spikeOut, gt, numSpikesScale=1):
        '''
        Calculates spike loss based on number of spikes within a `target region`.
        The `target region` and `desired spike count` is specified in ``error.errorDescriptor['tgtSpikeRegion']``
        Any spikes outside the target region are penalized with ``error.spikeTime`` loss..

        .. math::
            e(t) &= 
            \\begin{cases}
            \\frac{acutalSpikeCount - desiredSpikeCount}{targetRegionLength} & \\text{for }t \in targetRegion\\\\
            \\left(\\varepsilon * (output - desired)\\right)(t) & \\text{otherwise}
            \\end{cases}
            
            E &= \\int_0^T e(t)^2 \\text{d}t

        Arguments:
            * ``spikeOut`` (``torch.tensor``): spike tensor
            * ``desiredClass`` (``torch.tensor``): one-hot encoded desired class tensor. Time dimension should be 1 and rest of the tensor dimensions should be same as ``spikeOut``.

        Usage:

        >>> loss = error.numSpikes(spikeOut, target)
        '''
        # Tested with autograd, it works
        # Tested with autograd, it works
        # assert self.errorDescriptor['type'] == 'NumSpikes', "Error type is not NumSpikes"
        # desiredClass should be one-hot tensor with 5th dimension 1
        tgtSpikeRegion = self.errorDescriptor['tgtSpikeRegion']
        tgtSpikeCount  = self.errorDescriptor['tgtSpikeCount']
        startID = np.rint( tgtSpikeRegion['start'] / self.simulation['Ts'] ).astype(int)
        stopID  = np.rint( tgtSpikeRegion['stop' ] / self.simulation['Ts'] ).astype(int)
        
        actualSpikes = torch.sum(spikeOut[...,startID:stopID], 4, keepdim=True).cpu().detach().numpy() * self.simulation['Ts']
        desiredSpikes = gt[...,startID:stopID]
        errorSpikeCount = (actualSpikes - desiredSpikes) / (stopID - startID) * numSpikesScale
        targetRegion = np.zeros(spikeOut.shape)
        targetRegion[:,:,:,:,startID:stopID] = 1;
        spikeDesired = torch.FloatTensor(targetRegion * spikeOut.cpu().data.numpy()).to(spikeOut.device)
        
        error = self.slayer.psp(spikeOut - spikeDesired)

        error += torch.FloatTensor(errorSpikeCount * targetRegion).to(spikeOut.device)

        return 1/2 * torch.sum(error**2) * self.simulation['Ts']

    def MSE(self, spikeOut, spikeDesired):

        return torch.mean(torch.sqrt(torch.sum((spikeOut - spikeDesired) ** 2, dim=1)))

    def getIOU(self, spike_pred, spike_gt):
        spike_pred = spike_pred.cpu().numpy()
        spike_gt = spike_gt.cpu().numpy()

        intersection = np.sum(np.logical_and(spike_pred, spike_gt))
        union = np.sum(np.logical_or(spike_pred, spike_gt))
        return intersection/union
