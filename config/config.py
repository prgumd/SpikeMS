import os
import torch

from strictyaml import load, Map, Str, Int, Float, EmptyNone


class TestConfig:
    _str_to_dtype = {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'uint8': torch.uint8,
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
    }
    _schema = Map({
        'spikeConversion': Map({
            'intensity': Int(),                                
        }),
        'simulation': Map({
            'Ts': Float(),                                # Time-discretization in seconds
            'tSample': Int(),                           # Number of simulation steps
            'tStartLoss': Int(),                        # Start computing loss at this time-step
        }),
        'data': Map({
            'height': Int(),                              # full height of input
            'width': Int(),                          # full width of input
            'height_c': Int(),                  #if cropping, height to crop to
            'width_c': Int(),                  #if cropping, width to crop to
            'k': Int(),                        #number of 2D images of events used per frame
            'minEvents': Int(),               #minimum number of events to use for a frame to be included in dataset
            'start': Int(), ## some mask images start numbering images with an index other than 1
            'timePerMask': Float(),                                # Used to calculate mask time for datasets such as MVEC which provide masks at periodic intervals. In seconds
        }),
        'batchsize': Int(),
        'hardware': Map({
            'readerThreads': EmptyNone() | Int(),       # {empty: cpu_count, 0: main thread, >0: num threads used}
            'gpuDevice': Int(),                         # GPU to be used by device number
        }),
        'model': Map({
            'dtype': EmptyNone() | Int(),       # {empty: cpu_count, 0: main thread, >0: num threads used}
            'testSplit': Float(),               # ratio of data used for testing
        }),
        'neuron': Map({
            'type': Str(),                              # {cnn5-avgp-fc1}
            'theta': Float(),                          # Path to checkpoint
            'tauSr': Float(),               # {float16, float32, float64, uint8, int8, int16, int32, int64}
            'tauRef': Float(),               # {float16, float32, float64, uint8, int8, int16, int32, int64}
            'scaleRef': Int(),               # {float16, float32, float64, uint8, int8, int16, int32, int64}
            'tauRho': Int(),               # {float16, float32, float64, uint8, int8, int16, int32, int64}
            'scaleRho': Float(),               # {float16, float32, float64, uint8, int8, int16, int32, int64}
        }),
    })

    def __init__(self, config_filepath):
        with open(config_filepath, 'r') as stream:
            self.dictionary = load(stream.read(), self._schema).data


        # Some sanity checks.
        assert self.dictionary['simulation']['tSample'] > self.dictionary['simulation']['tStartLoss']
        assert self.dictionary['batchsize'] >= 1

        model_dtype_str = self.dictionary['model']['dtype']
        if model_dtype_str is None:
            self.dictionary['model']['dtype'] = torch.float32
        else:
            self.dictionary['model']['dtype'] = self._str_to_dtype[model_dtype_str]

        self.dictionary['hardware']['gpuDevice'] = torch.device('cuda:{}'.format(
            self.dictionary['hardware']['gpuDevice']))

        if self.dictionary['hardware']['readerThreads'] is None:
            self.dictionary['hardware']['readerThreads'] = os.cpu_count()

    def __getitem__(self, key):
        return self.dictionary[key]

    def __setitem__(self, key, value):
        self.dictionary[key] = value
