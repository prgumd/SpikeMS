import numpy as np
import  h5py
import os
from pathlib import Path
import torch
from torch.utils.data.dataset import Dataset
import slayerpytorch as snn

from torchvision import transforms
from PIL import Image

from .utils import SpikeRepresentationGenerator
from matplotlib import pyplot as plt

import cv2

class EVIMODatasetBase(Dataset):
    def __init__(self, datafile, genconfigs, maskDir, crop, maxBackgroundRatio, incrementPercent):

        self.height = genconfigs['data']['height']
        self.width = genconfigs['data']['width'] 
        self.crop = crop
        self.maxBackgroundRatio = maxBackgroundRatio
        
        if (self.crop):
            self.height_c = genconfigs['data']['height_c'] 
            self.width_c = genconfigs['data']['width_c'] 

        self.k = genconfigs['data']['k']
        self.min_events = genconfigs['data']['minEvents'] 
        self.start = genconfigs['data']['start']
        
        self.num_time_bins = genconfigs['simulation']['tSample'] 

        self.maskDir = maskDir

        self.data = h5py.File(datafile, 'r')
        self.length =  len(self.data['events_idx']) - self.k - 1 
        print("EVIMO tot length", self.length)

        self.incrementPercent = incrementPercent

    def getHeightAndWidth(self):
        assert self.height
        assert self.width
        return self.height, self.width

    @staticmethod
    def isDataFile(filepath: str):
        suffix = Path(filepath).suffix
        return suffix == '.h5' or suffix == '.npz'

    def __len__(self):
        return self.length

    def _preprocess(self, events, start, stop):
        return self._collate(events, start, stop)

    def get_event_idxs(self, index):
        return self.data['events_idx'][index], self.data['events_idx'][index+self.k] - 1


    def __getitem__(self, index: int):
        idx0, idx1 = self.get_event_idxs(index)
    
        events = self.data['events'][idx0:idx1]

        start_t = self.data['timeframes'][index][0]
        stop_t = self.data['timeframes'][index+self.k-1][1]
        
        if(self.incrementPercent != 1):
            stop_t =  start_t + (stop_t - start_t)* self.incrementPercent
            events = events[events[:,0] < stop_t,:]

        ts = events[:,0]
        xs = events[:,1] 
        ys = events[:,2]
        ps = events[:,3]

        # print("index {} idx0 {} idx1 {} k {} \n time {}, {} start_t {} stop_t {}\n".format(
        #     index, idx0, idx1, self.k, ts[0], ts[-1], start_t, stop_t))
 
        spike_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))   
        ts = (self.num_time_bins-1) * (ts - start_t) /(stop_t - start_t)
        
        spike_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))   
        spike_tensor[ps, ys, xs, ts] = 1

        full_mask_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))   

        for i in range(0, self.k):
            curr_start = int((self.num_time_bins) * (i)/self.k)
            curr_end = int((self.num_time_bins) * (i+1)/self.k)

            currfile_nm = os.path.join(self.maskDir, "depth_mask_{:d}.png".format(
                int(self.data['timeframes'][index][2])))

            if (not os.path.isfile(currfile_nm)):
                print("mask file not found", currfile_nm)
                return None
            fullmask = np.asarray(Image.open(currfile_nm))[:,:,0]
            fullmask = fullmask.astype(bool).astype(float)

            if (np.sum(fullmask) < self.min_events):
                print("not enough events", np.sum(fullmask), " < min events: ", self.min_events)
                return None

            kernel = np.ones((5, 5), 'uint8')
            fullmask = cv2.dilate(fullmask, kernel, iterations=1)

            fullmask = np.expand_dims(fullmask, axis=(0,3))
            tiled = torch.from_numpy(np.tile(fullmask, (2,1,1, curr_end-curr_start)))      

            full_mask_tensor[:,:,:,curr_start:curr_end] = tiled  

        masked_spike_tensor = ((spike_tensor + full_mask_tensor) > 1).float()
        background_spikes = (spike_tensor + torch.logical_not(masked_spike_tensor).float()) > 1


        if (self.crop):
            summed = torch.sum(masked_spike_tensor, axis = (0,3))
            max_v = torch.argmax(summed)
            
            center_x = int(max_v % self.width)
            center_y = int(max_v / self.width)

            crop_x_min  = int(max(center_x - int(self.width_c/2), 0))
            crop_y_min  = int(max(center_y - int(self.height_c/2), 0))

            if ((center_x + int(self.width_c/2)) > self.width - 1):
                crop_x_min = self.width - 1 - self.width_c

            if ((center_y + int(self.height_c/2)) > self.height - 1):
                crop_y_min = self.height - 1 - self.height_c


            spike_tensor = spike_tensor[:,
                        int(crop_y_min):int(crop_y_min+self.height_c), 
                        int(crop_x_min):int(crop_x_min+self.width_c), 
                        :]
            masked_spike_tensor = masked_spike_tensor[:,
                                int(crop_y_min):int(crop_y_min+self.height_c),
                                int(crop_x_min):int(crop_x_min+self.width_c), :100]
            full_mask_tensor = full_mask_tensor[:,
                                int(crop_y_min):int(crop_y_min+self.height_c),
                                int(crop_x_min):int(crop_x_min+self.width_c), :100]

        if (torch.sum(background_spikes)/torch.sum(masked_spike_tensor) > self.maxBackgroundRatio):
            return None

        assert not torch.isnan(spike_tensor).any()
        assert not torch.isnan(masked_spike_tensor).any()
        out = {
            'file_number': index,
            'spike_tensor': spike_tensor,
            'masked_spike_tensor': masked_spike_tensor,
            'full_mask_tensor': full_mask_tensor,
            'time_start': start_t,
            'time_per_index': (stop_t - start_t)/self.num_time_bins,
            'ratio': torch.sum(background_spikes)/torch.sum(masked_spike_tensor)
        }
        return out

class mvecDatasetBase(Dataset):
    def __init__(self, data_file, genconfigs):

        self.height = genconfigs['data']['height']
        self.width = genconfigs['data']['width'] 
        self.k = genconfigs['data']['k']
        self.num_time_bins = genconfigs['simulation']['tSample'] 

        self.data = h5py.File(data_file, 'r')['davis']['left']
        self.length = self.data['image_raw_ts'].shape[0]-self.k

        print("length is", self.data['image_raw_ts'].shape)

    def getHeightAndWidth(self):
        assert self.height
        assert self.width
        return self.height, self.width

    @staticmethod
    def isDataFile(filepath: str):
        suffix = Path(filepath).suffix
        return suffix == '.h5' or suffix == '.npz'

    def __len__(self):
        return self.length

    def get_event_idxs(self, index, k):
        return self.data['image_raw_event_inds'][index], self.data['image_raw_event_inds'][index+k]

    def get_start_stop(self, index, k):
        return self.data['image_raw_ts'][index], self.data['image_raw_ts'][index+k]
        
    def __getitem__(self, index: int):        
        idx0, idx1 = self.get_event_idxs(index, self.k)
        if(idx1-idx0)<2000:
           return None
        start, stop = self.get_start_stop(index, self.k)       
        events = self.data['events'][idx0:idx1]
    
        #event frame
        ys = events[:,0]
        xs = events[:,1]
        ts = events[:,2]
        ps = events[:,3]

        ts_bins = (self.num_time_bins-1)*(ts - start) / (stop - start) #normalize time
        ps_shift = (ps+1)/2 # (-1, 1) -> (0, 1)


        binary_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))   
        binary_tensor[ps, xs, ys, ts_bins] = 1

        events[:,2] = ts_bins
        events[:,3] = ps_shift

        spike_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))   
        uniq, num_e = torch.unique(torch.tensor(events), return_counts=True, dim=0)
        maxcount = torch.max(num_e)
        num_e = num_e.float()/maxcount

        uniq = torch.round(uniq).numpy()

        ys_u = uniq[:,0]
        xs_u = uniq[:,1] 
        ts_u = uniq[:,2]
        ps_u = uniq[:,3]

        spike_tensor[ps_u, xs_u, ys_u, ts_u] = num_e
        assert not torch.isnan(spike_tensor).any()

        out = {
            'file_number': index,
            'spike_tensor': spike_tensor,
            'binary_tensor': binary_tensor
        }
        return out

class MODDatasetBase(Dataset):
    def __init__(self, datafile: str, genconfigs, maskDir, crop, maxBackgroundRatio, incrementalPercent):

        self.height = genconfigs['data']['height']
        self.width = genconfigs['data']['width'] 
        self.k = genconfigs['data']['k'] 
        self.increment = genconfigs['data']['timePerMask'] 
        self.counter = 0
        self.data = h5py.File(datafile, 'r')
        self.length = self.data['images_idx'].shape[0] -1 - self.k
        self.num_time_bins = genconfigs['simulation']['tSample'] 

        self.maskDir = maskDir
        self.incrementalPercent = incrementalPercent
        self.min_events = genconfigs['data']['minEvents'] 
        self.maxBackgroundRatio = maxBackgroundRatio

    def getHeightAndWidth(self):
        assert self.height
        assert self.width
        return self.height, self.width

    @staticmethod
    def isDataFile(filepath: str):
        suffix = Path(filepath).suffix
        return suffix == '.h5' or suffix == '.npz'

    def __len__(self):
        return self.length

    def _preprocess(self, events, start, stop):
        return self._collate(events, start, stop)

    def get_event_idxs(self, index, k):
        return self.data['images_idx'][index], self.data['images_idx'][index+k]

    def get_start_stop(self, index, k):
        return self.increment*(index+1), self.increment*(index + 1 + k)

    def get_maskedevent_idxs(self, index, k):
        return self.data['masked_event_idx'][index], self.data['masked_event_idx'][index+k]

    def __getitem__(self, index: int):        

        start_t, stop_t = self.get_start_stop(index, self.k)    

        idx0, idx1 = self.get_event_idxs(index, self.k)
        events = self.data['events'][idx0:idx1]

        midx0, midx1 = self.get_maskedevent_idxs(index, self.k)
        maskedevents = self.data['masked_events'][midx0:midx1]

        
        if(self.incrementalPercent < 1):
            stop_t =  start_t + (stop_t - start_t)*self.incrementalPercent
            events = events[events[:,0] < stop_t,:]
            maskedevents = maskedevents[maskedevents[:,0] < stop_t,:]

        ts = events[:,0]
        xs = events[:,1] 
        ys = events[:,2]
        ps = events[:,3]

        m_ts = maskedevents[:,0]
        m_xs = maskedevents[:,1] 
        m_ys = maskedevents[:,2]
        m_ps = maskedevents[:,3]
        rat = len(m_ps)
        
        if (len(m_ps) > 0):
            rat = len(ps)/len(m_ps)

        ts = (self.num_time_bins-1) * (ts - start_t) /(stop_t - start_t)
        m_ts = (self.num_time_bins-1) * (m_ts - start_t) /(stop_t - start_t)
        
        spike_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))   
        spike_tensor[ps, ys, xs, ts] = 1
        
        masked_spike_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))   
        masked_spike_tensor[m_ps, m_ys, m_xs, m_ts] = 1
        full_mask_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))   

        for i in range(0, self.k):
            curr_start = int((self.num_time_bins) * (i)/self.k)
            curr_end = int((self.num_time_bins) * (i+1)/self.k)

            curr_masknm = os.path.join(self.maskDir, "mask_{:08d}.png".format(index+i+1))
            fullmask = np.asarray(Image.open(curr_masknm))
            fullmask = np.expand_dims(fullmask, axis=(0,3))
            tiled = torch.from_numpy(np.tile(fullmask, (2,1,1, curr_end-curr_start)))        

            full_mask_tensor[:,:,:,curr_start:curr_end] = tiled  

        #TODO when time bins not divisible by k
        # sample = self.samples[index]
        # ex_imsize = self._extend_size(self.imsize)
        # img1 = self.load_as_float(sample['img1'], ex_imsize)
        # img2 = self.load_as_float(sample['img2'], ex_imsize)
        # start = sample['start']
        # stop = sample['stop']
        # frame_events  = self.compute_event_image(*self._preprocess([sample['frame_events']], [start], [stop]), ex_imsize)
        # frame_events = frame_events[0]

        #larger allowed ratio decreases quality of model but utilizes more events
        if (len(m_ps) == 0 or len(ps)/len(m_ps) > self.maxBackgroundRatio):
             return None

        assert not torch.isnan(spike_tensor).any()
        assert not torch.isnan(masked_spike_tensor).any()
        out = {
            'file_number': index+1,
            'time_start': start_t,
            'time_per_index': (stop_t - start_t)/self.num_time_bins,
            'spike_tensor': spike_tensor,
            'masked_spike_tensor': masked_spike_tensor,
            'full_mask_tensor': full_mask_tensor,
            'ratio': len(ps)/len(m_ps)
        }
        return out     

