import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import slayerCuda

import sparseconvnet as scn



def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

class spikeLayer(torch.nn.Module):
    '''
    This class defines the main engine of SLAYER.
    It provides necessary functions for describing a SNN layer.
    The input to output connection can be fully-connected, convolutional, or aggregation (pool)
    It also defines the psp operation and spiking mechanism of a spiking neuron in the layer.

    **Important:** It assumes all the tensors that are being processed are 5 dimensional. 
    (Batch, Channels, Height, Width, Time) or ``NCHWT`` format.
    The user must make sure that an input of correct dimension is supplied.

    *If the layer does not have spatial dimension, the neurons can be distributed along either
    Channel, Height or Width dimension where Channel * Height * Width is equal to number of neurons.
    It is recommended (for speed reasons) to define the neuons in Channels dimension and make Height and Width
    dimension one.*

    Arguments:
        * ``neuronDesc`` (``slayerParams.yamlParams``): spiking neuron descriptor.
            .. code-block:: python

                neuron:
                    type:     SRMALPHA  # neuron type
                    theta:    10    # neuron threshold
                    tauSr:    10.0  # neuron time constant
                    tauRef:   1.0   # neuron refractory time constant
                    scaleRef: 2     # neuron refractory response scaling (relative to theta)
                    tauRho:   1     # spike function derivative time constant (relative to theta)
                    scaleRho: 1     # spike function derivative scale factor
        * ``simulationDesc`` (``slayerParams.yamlParams``): simulation descriptor
            .. code-block:: python

                simulation:
                    Ts: 1.0         # sampling time (ms)
                    tSample: 300    # time length of sample (ms)   
        * ``fullRefKernel`` (``bool``, optional): high resolution refractory kernel (the user shall not use it in practice)  

    Usage:

    >>> snnLayer = slayer.spikeLayer(neuronDesc, simulationDesc)
    '''
    def __init__(self, neuronDesc, simulationDesc, fullRefKernel = False):
        super(spikeLayer, self).__init__()
        self.neuron = neuronDesc
        self.simulation = simulationDesc
        self.fullRefKernel = fullRefKernel

        self.register_buffer('srmKernel', self.calculateSrmKernel())
        self.register_buffer('refKernel', self.calculateCustomRefKernel())
        # self.register_buffer('refKernel', self.calculateRefKernel())
# 
    def calculateSrmKernel(self):

        srmKernel = self._calculateAlphaKernel(self.neuron['tauSr'])
        # return torch.tensor(srmKernel)
        return torch.FloatTensor(srmKernel)

    def calculateRefKernel(self):
        if self.fullRefKernel:
            refKernel = self._calculateAlphaKernel(tau=self.neuron['tauRef'], mult = -self.neuron['scaleRef'] * self.neuron['theta'], EPSILON = 0.0001)
            # This gives the high precision refractory kernel as MATLAB implementation, however, it is expensive
        else:
            refKernel = self._calculateAlphaKernel(tau=self.neuron['tauRef'], mult = -self.neuron['scaleRef'] * self.neuron['theta'])

        # return torch.tensor(refKernel)
        return torch.FloatTensor(refKernel)

    def calculateCustomRefKernel(self, EPSILON: float = 0.01):
        tau = self.neuron['tauRef']
        mult = - self.neuron['scaleRef'] * self.neuron['theta']
        assert tau > 0.
        assert mult < 0.
        time = np.arange(0, self.simulation['tSample'], self.simulation['Ts'])
        potential = mult * np.exp(-1 / tau * time[:-1])
        return torch.from_numpy(np.concatenate((np.array([0]), potential[np.abs(potential) > EPSILON]))).float()

    def _calculateAlphaKernel(self, tau, mult = 1, EPSILON = 0.01):
        eps = []

        for t in np.arange(0, self.simulation['tSample'], self.simulation['Ts']):
            epsVal = mult * t / tau * math.exp(1 - t / tau)
            if abs(epsVal) < EPSILON and t > tau:
                break
            eps.append(epsVal)
        return eps

    def _zeroPadAndFlip(self, kernel):
        if (len(kernel)%2) == 0: kernel.append(0)
        prependedZeros = np.zeros((len(kernel) - 1))
        return np.flip( np.concatenate( (prependedZeros, kernel) ) ).tolist()

    def psp(self, spike):
        '''
        Applies psp filtering to spikes.
        The output tensor dimension is same as input.

        Arguments:
            * ``spike``: input spike tensor.

        Usage:

        >>> filteredSpike = snnLayer.psp(spike)
        '''
        return _pspFunction.apply(spike, self.srmKernel, self.simulation['Ts'])

    def pspLayer(self):
        '''
        Returns a function that can be called to apply psp filtering to spikes.
        The output tensor dimension is same as input.
        The initial psp filter corresponds to the neuron psp filter.
        The psp filter is learnable.
        NOTE: the learned psp filter must be reversed because PyTorch performs conrrelation operation.

        Usage:

        >>> pspLayer = snnLayer.pspLayer()
        >>> filteredSpike = pspLayer(spike)
        '''
        return _pspLayer(self.srmKernel, self.simulation['Ts'])

    # def pspFilter(self, nFilter, filterLength, filterScale=1):
    #     '''
    #     Returns a function that can be called to apply a bank of temporal filters.
    #     The output tensor is of same dimension as input except the channel dimension is scaled by number of filters.
    #     The initial filters are initialized using default PyTorch initializaion for conv layer.
    #     The filter banks are learnable.
    #     NOTE: the learned psp filter must be reversed because PyTorch performs conrrelation operation.

    #     Arguments:
    #         * ``nFilter``: number of filters in the filterbank.
    #         * ``filterLength``: length of filter in number of time bins.
    #         * ``filterScale``: initial scaling factor for filter banks. Default: 1.

    #     Usage:

    #     >>> pspFilter = snnLayer.pspFilter()
    #     >>> filteredSpike = pspFilter(spike)
    #     '''
    #     return _pspFilter(nFilter, filterLength, self.simulation['Ts'], filterScale)

    def replicateInTime(self, input, mode='nearest'):
        Ns = int(self.simulation['tSample'] / self.simulation['Ts'])
        N, C, H, W = input.shape
        # output = F.pad(input.reshape(N, C, H, W, 1), pad=(Ns-1, 0, 0, 0, 0, 0), mode='replicate')
        if mode == 'nearest':
            output = F.interpolate(input.reshape(N, C, H, W, 1), size=(H, W, Ns), mode='nearest')
        return output

    def dense(self, inFeatures, outFeatures, weightScale=10):   # default weight scaling of 10
        '''
        Returns a function that can be called to apply dense layer mapping to input tensor per time instance.
        It behaves similar to ``torch.nn.Linear`` applied for each time instance.

        Arguments:
            * ``inFeatures`` (``int``, tuple of two ints, tuple of three ints): 
              dimension of input features (Width, Height, Channel) that represents the number of input neurons.
            * ``outFeatures`` (``int``): number of output neurons.
            * ``weightScale``: sale factor of default initialized weights. Default: 10

        Usage:

        >>> fcl = snnLayer.dense(2048, 512)          # takes (N, 2048, 1, 1, T) tensor
        >>> fcl = snnLayer.dense((128, 128, 2), 512) # takes (N, 2, 128, 128, T) tensor
        >>> output = fcl(input)                      # output will be (N, 512, 1, 1, T) tensor
        '''
        return _denseLayer(inFeatures, outFeatures, weightScale)    

    def conv(self, inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=100):    # default weight scaling of 100
        '''
        Returns a function that can be called to apply conv layer mapping to input tensor per time instance.
        It behaves same as ``torch.nn.conv2d`` applied for each time instance.

        Arguments:
            * ``inChannels`` (``int``): number of channels in input
            * ``outChannels`` (``int``): number of channls produced by convoluion
            * ``kernelSize`` (``int`` or tuple of two ints): size of the convolving kernel
            * ``stride`` (``int`` or tuple of two ints): stride of the convolution. Default: 1
            * ``padding`` (``int`` or tuple of two ints):   zero-padding added to both sides of the input. Default: 0
            * ``dilation`` (``int`` or tuple of two ints): spacing between kernel elements. Default: 1
            * ``groups`` (``int`` or tuple of two ints): number of blocked connections from input channels to output channels. Default: 1
            * ``weightScale``: sale factor of default initialized weights. Default: 100

        The parameters ``kernelSize``, ``stride``, ``padding``, ``dilation`` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

        Usage:

        >>> conv = snnLayer.conv(2, 32, 5) # 32C5 flter
        >>> output = conv(input)           # must have 2 channels
        '''


        # return _convLayer(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale) 

        kernel_ = (kernelSize, kernelSize, 1)
        stride_ = (stride, stride, 1)
        padding_ = (padding, padding, 0)
        dilation_ = (dilation, dilation, 1)

        # return nn.Conv3d(inChannels, outChannels, kernel_size=kernel_, stride=stride_, padding=padding_, dilation=dilation_, groups=1, bias=False)
        return scn.Convolution(3, inChannels, outChannels, kernelSize, stride, False, groups=1)
        # return nn.Sequential(
        # nn.Conv3d(inChannels, outChannels, kernel_size=kernel_, stride=stride_, padding=padding_, dilation=dilation_, groups=1, bias=False),
        # # nn.BatchNorm3d(outChannels)
        # # nn.LeakyReLU(0.1, inplace=True)
        # )

    def deconv(self, inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=100):    # default weight scaling of 100 
        # inFeatures = inChannels
        # outFeatures =  outChannels

        kernel_ = (kernelSize, kernelSize, 1)
        stride_ = (stride, stride, 1)
        padding_ = (padding, padding, 0)
        dilation_ = (dilation, dilation, 1)

        return scn.Deconvolution(3, inChannels, outChannels, kernelSize, stride, False, groups=1)
       # return nn.ConvTranspose3d(inChannels, outChannels, kernel_size=kernel_, stride=stride_, padding=padding_, dilation=dilation_, groups=1, bias=False)
        # return nn.Sequential(
        #    nn.ConvTranspose3d(inChannels, outChannels, kernel_size=kernel_, stride=stride_, padding=padding_, dilation=dilation_, groups=1, bias=False),
        #    # nn.BatchNorm3d(outChannels)
        #    # nn.LeakyReLU(0.1, inplace=True)
        #    )
  
        # return _deconvLayer(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale) 



    def pool(self, kernelSize, stride=None, padding=0, dilation=1):
        '''
        Returns a function that can be called to apply pool layer mapping to input tensor per time instance.
        It behaves same as ``torch.nn.``:sum pooling applied for each time instance.

        Arguments:
            * ``kernelSize`` (``int`` or tuple of two ints): the size of the window to pool over
            * ``stride`` (``int`` or tuple of two ints): stride of the window. Default: `kernelSize`
            * ``padding`` (``int`` or tuple of two ints): implicit zero padding to be added on both sides. Default: 0
            * ``dilation`` (``int`` or tuple of two ints): a parameter that controls the stride of elements in the window. Default: 1

        The parameters ``kernelSize``, ``stride``, ``padding``, ``dilation`` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

        Usage:

        >>> pool = snnLayer.pool(4) # 4x4 pooling
        >>> output = pool(input)
 
        '''
        kernel = (kernelSize[0], kernelSize[1], 1)
        stride = (stride[0], stride[1], 1)
        padding = (padding[0], padding[1], 0)
        dilation = (dilation[0], dilation[1], 1)

        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False)
 #       return _poolLayer(self.neuron['theta'], kernelSize, stride, padding, dilation)

    def dropout(self, p=0.5, inplace=False):
        '''
        Returns a function that can be called to apply dropout layer to the input tensor.
        It behaves similar to ``torch.nn.Dropout``.
        However, dropout over time dimension is preserved, i.e.
        if a neuron is dropped, it remains dropped for entire time duration.

        Arguments:
            * ``p``: dropout probability.
            * ``inplace`` (``bool``): inplace opeartion flag.

        Usage:

        >>> drop = snnLayer.dropout(0.2)
        >>> output = drop(input)
        '''
        return _dropoutLayer(p, inplace)

    def delayShift(self, input, delay, Ts=1):
        '''
        Applies delay in time dimension (assumed to be the last dimension of the tensor) of the input tensor.
        The autograd backward link is established as well.

        Arguments:
            * ``input``: input Torch tensor.
            * ``delay`` (``float`` or Torch tensor): amount of delay to apply.
              Same delay is applied to all the inputs if ``delay`` is ``float`` or Torch tensor of size 1.
              If the Torch tensor has size more than 1, its dimension  must match the dimension of input tensor except the last dimension.
            * ``Ts``: sampling time of the delay. Default is 1.

        Usage:

        >>> delayedInput = slayer.delayShift(input, 5)
        '''
        return _delayFunctionNoGradient.apply(input, delay, Ts)

    def delay(self, inputSize):
        '''
        Returns a function that can be called to apply delay opeartion in time dimension of the input tensor.
        The delay parameter is available as ``delay.delay`` and is initialized uniformly between 0ms  and 1ms.
        The delay parameter is stored as float values, however, it is floored during actual delay applicaiton internally.
        The delay values are not clamped to zero.
        To maintain the causality of the network, one should clamp the delay values explicitly to ensure positive delays.

        Arguments:
            * ``inputSize`` (``int`` or tuple of three ints): spatial shape of the input signal in CHW format (Channel, Height, Width).
              If integer value is supplied, it refers to the number of neurons in channel dimension. Heighe and Width are assumed to be 1.   

        Usage:

        >>> delay = snnLayer.delay((C, H, W))
        >>> delayedSignal = delay(input)

        Always clamp the delay after ``optimizer.step()``.

        >>> optimizer.step()
        >>> delay.delay.data.clamp_(0)  
        '''
        return _delayLayer(inputSize, self.simulation['Ts'])

    def spike(self, membranePotential):
        '''
        Applies spike function and refractory response.
        The output tensor dimension is same as input.
        ``membranePotential`` will reflect spike and refractory behaviour as well.

        Arguments:
            * ``membranePotential``: subthreshold membrane potential.

        Usage:

        >>> outSpike = snnLayer.spike(membranePotential)
        '''
        return _spikeFunction.apply(membranePotential, self.refKernel, self.neuron, self.simulation['Ts'])


class _denseLayer(nn.Conv3d):
    def __init__(self, inFeatures, outFeatures, weightScale=1):
        # extract information for kernel and inChannels
        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures 
        elif len(inFeatures) == 2:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = 1
        elif len(inFeatures) == 3:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = inFeatures[2]
        else:
            raise Exception('inFeatures should not be more than 3 dimension. It was: {}'.format(inFeatures.shape))
        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(outFeatures.shape))
        super(_denseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)

        if weightScale != 1:    
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed

    def forward(self, input):
        return F.conv3d(input, 
                        self.weight, self.bias, 
                        self.stride, self.padding, self.dilation, self.groups)


class _convLayer(nn.Conv3d):
    def __init__(self, inFeatures, outFeatures, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=1):
        inChannels = inFeatures
        outChannels = outFeatures

        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        super(_convLayer, self).__init__(inChannels, outChannels, kernel, stride, padding, dilation, groups, bias=False)

        if weightScale != 1:    
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
  #      print("conv weight shape", self.weight.shape)
    def foward(self, input):
        return F.conv3d(input, 
                        self.weight, self.bias, 
                        self.stride, self.padding, self.dilation, self.groups)


class _poolLayer(nn.Conv3d):
    def __init__(self, theta, kernelSize, stride=None, padding=0, dilation=1):
        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))

        # stride
        if stride is None:
            stride = kernel
        elif type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))
        super(_poolLayer, self).__init__(1, 1, kernel, stride, padding, dilation, bias=False)   

        # set the weights to 1.1*theta and requires_grad = False
        self.weight = torch.nn.Parameter(torch.FloatTensor(1.1 * theta * np.ones((self.weight.shape))).to(self.weight.device), requires_grad = False)


    def forward(self, input):
        device = input.device
        dtype  = input.dtype

        if input.shape[2]%self.weight.shape[2] != 0:
            input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2]%self.weight.shape[2], input.shape[3], input.shape[4]), dtype=dtype).to(device)), 2)
        if input.shape[3]%self.weight.shape[3] != 0:
            input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3]%self.weight.shape[3], input.shape[4]), dtype=dtype).to(device)), 3)

        dataShape = input.shape

        result = F.conv3d(input.reshape((dataShape[0], 1, dataShape[1] * dataShape[2], dataShape[3], dataShape[4])), 
                          self.weight, self.bias, 
                          self.stride, self.padding, self.dilation)

        return result.reshape((result.shape[0], dataShape[1], -1, result.shape[3], result.shape[4]))


class _dropoutLayer(nn.Dropout3d):
    def forward(self, input):
        inputShape = input.shape
        return F.dropout3d(input.reshape((inputShape[0], -1, 1, 1, inputShape[-1])),
                           self.p, self.training, self.inplace).reshape(inputShape)


class _pspLayer(nn.Conv3d):
    def __init__(self, filter, Ts):
        inChannels  = 1
        outChannels = 1
        kernel      = (1, 1, torch.numel(filter))

        self.Ts = Ts

        super(_pspLayer, self).__init__(inChannels, outChannels, kernel, bias=False) 

        flippedFilter = torch.tensor(np.flip(filter.cpu().data.numpy()).copy()).reshape(self.weight.shape)

        self.weight = torch.nn.Parameter(flippedFilter.to(self.weight.device), requires_grad = True)

        self.pad = torch.nn.ConstantPad3d(padding=(torch.numel(filter)-1, 0, 0, 0, 0, 0), value=0)

    def forward(self, input):
        # inShape = input.shape
        # inPadded = self.pad(input.reshape((inShape[0], 1, 1, -1, inShape[-1])))
        # output = F.conv3d(inPadded, self.weight) * self.Ts

        # return output.reshape(inShape)
        input("bad pspLayer")
        return None


class _pspFilter(nn.Conv3d):
    def __init__(self, nFilter, filterLength, Ts, filterScale=1):
        inChannels  = 1
        outChannels = nFilter
        kernel      = (1, 1, filterLength)

        super(_pspFilter, self).__init__(inChannels, outChannels, kernel, bias=False) 

        self.Ts  = Ts
        self.pad = torch.nn.ConstantPad3d(padding=(filterLength-1, 0, 0, 0, 0, 0), value=0)

        if filterScale != 1:
            self.weight.data *= filterScale

    def forward(self, input):
        N, C, H, W, Ns = input.shape
        inPadded = self.pad(input.reshape((N, 1, 1, -1, Ns)))
        output = F.conv3d(inPadded, self.weight) * self.Ts
        return output.reshape((N, -1, H, W, Ns))

class _spikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membranePotential, refractoryResponse, neuron, Ts):
        device = membranePotential.device
        dtype  = membranePotential.dtype
        threshold      = neuron['theta']
        oldDevice = torch.cuda.current_device()
        spikes = slayerCuda.getSpikes(membranePotential, refractoryResponse, threshold, Ts)

        pdfScale        = torch.autograd.Variable(torch.tensor(neuron['scaleRho']                 , device=device, dtype=dtype), requires_grad=False)
        pdfTimeConstant = torch.autograd.Variable(torch.tensor(neuron['tauRho'] * neuron['theta'] , device=device, dtype=dtype), requires_grad=False) # needs to be scaled by theta
        threshold       = torch.autograd.Variable(torch.tensor(neuron['theta']                    , device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(membranePotential, threshold, pdfTimeConstant, pdfScale)
        return spikes

    @staticmethod
    def backward(ctx, gradOutput):
        (membranePotential, threshold, pdfTimeConstant, pdfScale) = ctx.saved_tensors
        reasonable = True
        sigmoid = False #modification using sigmoid vs PDF
        sigmoidScale = True #modification using sigmoid vs PDF

        if sigmoid:
            spikePdf = torch.exp(-(membranePotential - threshold))/ (1+torch.exp(-(membranePotential - threshold)))**2
        elif sigmoidScale:
            spikePdf = pdfScale * torch.exp(-(membranePotential - threshold))/ (1+torch.exp(-(membranePotential - threshold)))**2
        else:
            if reasonable:
                # For some reason the membrane potential clamping does not give good results.
                # membranePotential[membranePotential > threshold] = threshold
                # For some reason pdfScale should be scaled by the threshold to get decent results.
                spikePdf = pdfScale/threshold * torch.exp( -torch.abs(membranePotential - threshold) / pdfTimeConstant)
            else:
                spikePdf = pdfScale/pdfTimeConstant * torch.exp( -torch.abs(membranePotential - threshold) / pdfTimeConstant)

        return gradOutput * spikePdf, None, None, None

class _pspFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike, filter, Ts):
        device = spike.device
        dtype  = spike.dtype
        psp = slayerCuda.conv(spike.contiguous(), filter, Ts)
        Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(filter, Ts)
        return psp

    @staticmethod
    def backward(ctx, gradOutput):
        (filter, Ts) = ctx.saved_tensors
        gradInput = slayerCuda.corr(gradOutput.contiguous(), filter, Ts)
        if filter.requires_grad is False:
            gradFilter = None
        else:
            gradFilter = None
            pass
        return gradInput, gradFilter, None

class _pspSparseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike, filter, Ts):
        device = spike.device
        dtype  = spike.dtype
        psp = scn.spikeConvolution(3, inChannels, outChannels, kernelSize, stride, filter, False, groups=1)
        Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(filter, Ts)
        return psp

    @staticmethod
    def backward(ctx, gradOutput):
        (filter, Ts) = ctx.saved_tensors
        gradInput = slayerCuda.corr(gradOutput.contiguous(), filter, Ts)
        if filter.requires_grad is False:
            gradFilter = None
        else:
            gradFilter = None
            pass
        return gradInput, gradFilter, None

class _delayLayer(nn.Module):
    def __init__(self, inputSize, Ts):
        super(_delayLayer, self).__init__()

        if type(inputSize) == int:
            inputChannels = inputSize
            inputHeight   = 1
            inputWidth    = 1
        elif len(inputSize) == 3:
            inputChannels = inputSize[0]
            inputHeight   = inputSize[1]
            inputWidth    = inputSize[2]
        else:
            raise Exception('inputSize can only be 1 or 2 dimension. It was: {}'.format(inputSize.shape))

        self.delay = torch.nn.Parameter(torch.rand((inputChannels, inputHeight, inputWidth)), requires_grad=True)
        self.Ts = Ts

    def forward(self, input):
        N, C, H, W, Ns = input.shape
        if input.numel() != self.delay.numel() * input.shape[-1]:
            return _delayFunction.apply(input, self.delay.repeat((1, H, W)), self.Ts) # different delay per channel
        else:
            return _delayFunction.apply(input, self.delay, self.Ts)


class _delayFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, delay, Ts):
        device = input.device
        dtype  = input.dtype
        output = slayerCuda.shift(input, delay.data, Ts)
        Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(output, delay.data, Ts)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        (output, delay, Ts) = ctx.saved_tensors
        diffFilter = torch.tensor([-1, 1], dtype=gradOutput.dtype).to(gradOutput.device) / Ts
        outputDiff = slayerCuda.conv(output, diffFilter, 1)
        # the conv operation should not be scaled by Ts. 
        # As such, the output is -( x[k+1]/Ts - x[k]/Ts ) which is what we want.
        gradDelay  = torch.sum(gradOutput * outputDiff, [0, -1], keepdim=True).reshape(gradOutput.shape[1:-1]) * Ts
        # no minus needed here, as it is included in diffFilter which is -1 * [1, -1]

        return slayerCuda.shift(gradOutput, -delay, Ts), gradDelay, None


class _delayFunctionNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, delay, Ts=1):
        device = input.device
        dtype  = input.dtype
        output = slayerCuda.shift(input, delay, Ts)
        Ts     = torch.autograd.Variable(torch.tensor(Ts   , device=device, dtype=dtype), requires_grad=False)
        delay  = torch.autograd.Variable(torch.tensor(delay, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(delay, Ts)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        (delay, Ts) = ctx.saved_tensors
        return slayerCuda.shift(gradOutput, -delay, Ts), None, None

class _deconvLayer(nn.Conv3d):
    def __init__(self, inFeatures, outFeatures, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=1):
        inChannels = inFeatures
        outChannels = outFeatures
#        print("inChannels",inChannels)
#        print("outChannels", outChannels)
        
        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        super(_deconvLayer, self).__init__(outChannels, inChannels, kernel, stride, padding, dilation, groups, bias=False)

        if weightScale != 1:    
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
       # print("weight shape", self.weight.shape)

    def forward(self, input):
        dataShape = input.shape
        #print("dataShape",dataShape)
        return F.conv_transpose3d(input, 
                          self.weight, self.bias, 
                                self.stride, self.padding, self.output_padding, self.groups, self.dilation)
                          
  
#@weak_module
#class ConvTranspose2d(_ConvTransposeMixin, _ConvNd):
#     r"""Applies a 2D transposed convolution operator over an input image
#     composed of several input planes.

#     This module can be seen as the gradient of Conv2d with respect to its input.
#     It is also known as a fractionally-strided convolution or
#     a deconvolution (although it is not an actual deconvolution operation).

#     * :attr:`stride` controls the stride for the cross-correlation.

#     * :attr:`padding` controls the amount of implicit zero-paddings on both
#     sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
#     below for details.

#     * :attr:`output_padding` controls the additional size added to one side
#     of the output shape. See note below for details.

#     * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
#     It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

#     * :attr:`groups` controls the connections between inputs and outputs.
#     :attr:`in_channels` and :attr:`out_channels` must both be divisible by
#     :attr:`groups`. For example,

#         * At groups=1, all inputs are convolved to all outputs.
#         * At groups=2, the operation becomes equivalent to having two conv
#         layers side by side, each seeing half the input channels,
#         and producing half the output channels, and both subsequently
#         concatenated.
#         * At groups= :attr:`in_channels`, each input channel is convolved with
#         its own set of filters (of size
#         :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`).

#     The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
#     can either be:

#         - a single ``int`` -- in which case the same value is used for the height and width dimensions
#         - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
#         and the second `int` for the width dimension

#     .. note::

#         Depending of the size of your kernel, several (of the last)
#         columns of the input might be lost, because it is a valid `cross-correlation`_,
#         and not a full `cross-correlation`_.
#         It is up to the user to add proper padding.

#     .. note::
#         The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
#         amount of zero padding to both sizes of the input. This is set so that
#         when a :class:`~torch.nn.Conv2d` and a :class:`~torch.nn.ConvTranspose2d`
#         are initialized with same parameters, they are inverses of each other in
#         regard to the input and output shapes. However, when ``stride > 1``,
#         :class:`~torch.nn.Conv2d` maps multiple input shapes to the same output
#         shape. :attr:`output_padding` is provided to resolve this ambiguity by
#         effectively increasing the calculated output shape on one side. Note
#         that :attr:`output_padding` is only used to find output shape, but does
#         not actually add zero-padding to output.

#     .. include:: cudnn_deterministic.rst

#     Args:
#         in_channels (int): Number of channels in the input image
#         out_channels (int): Number of channels produced by the convolution
#         kernel_size (int or tuple): Size of the convolving kernel
#         stride (int or tuple, optional): Stride of the convolution. Default: 1
#         padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
#             will be added to both sides of each dimension in the input. Default: 0
#         output_padding (int or tuple, optional): Additional size added to one side
#             of each dimension in the output shape. Default: 0
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
#         bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
#         dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

#     Shape:
#         - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
#         - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

#         .. math::
#             H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
#                         \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1
#         .. math::
#             W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
#                         \times (\text{kernel\_size}[1] - 1) + \text{output\_padding}[1] + 1

#     Attributes:
#         weight (Tensor): the learnable weights of the module of shape
#                         :math:`(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}},`
#                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
#                         The values of these weights are sampled from
#                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
#                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
#         bias (Tensor):   the learnable bias of the module of shape (out_channels)
#                         If :attr:`bias` is ``True``, then the values of these weights are
#                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
#                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

#     Examples::

#         >>> # With square kernels and equal stride
#         >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
#         >>> # non-square kernels and unequal stride and with padding
#         >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
#         >>> input = torch.randn(20, 16, 50, 100)
#         >>> output = m(input)
#         >>> # exact output size can be also specified as an argument
#         >>> input = torch.randn(1, 16, 12, 12)
#         >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
#         >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
#         >>> h = downsample(input)
#         >>> h.size()
#         torch.Size([1, 16, 6, 6])
#         >>> output = upsample(h, output_size=input.size())
#         >>> output.size()
#         torch.Size([1, 16, 12, 12])

#     .. _cross-correlation:
#         https://en.wikipedia.org/wiki/Cross-correlation

#     .. _link:
#         https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
#     """

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                 padding=0, output_padding=0, groups=1, bias=True,
#                 dilation=1, padding_mode='zeros'):
#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         padding = _pair(padding)
#         dilation = _pair(dilation)
#         output_padding = _pair(output_padding)
#         super(ConvTranspose2d, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation,
#             True, output_padding, groups, bias, padding_mode)

 #    @weak_script_method
#     def forward(self, input, output_size=None):
         # type: (Tensor, Optional[List[int]]) -> Tensor
#         if self.padding_mode != 'zeros':
#             raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

#         output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

#         return F.conv_transpose2d(
#             input, self.weight, self.bias, self.stride, self.padding,
#             output_padding, self.groups, self.dilation)
