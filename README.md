# SpikeMS: Deep Spiking Neural Network for Motion Segmentation

This is the code for the paper **SpikeMS: Deep Spiking Neural Network for Motion Segmentation** by Chethan M. Parameshwara, Simin Li, Cornelia Fermuller, Nitin J. Sanket, Matthew S. Evanusa, Yiannis Aloimonos. 

<a href="http://prg.cs.umd.edu"><i>Perception & Robotics Group</i></a> at the Department of Computer Science, <a href="https://umd.edu/">University of Maryland- College Park</a>.

You can find a pdf of the paper [here](https://arxiv.org/pdf/2105.06562.pdf).

## Folders
- config - configuration processing and log dir creation
- dataloader - spike processing functions and pytorch dataloader (base.py). Handles incremental testing (Line 70) and filtering included in base.py (Line 148 to filter by ratio of background/object events)
- model - specifies SNN layer parameters utilizing pytorch/slayerpytorch. List of snn models mapped in utils.py 
- slayerpytorch - Modified version of [**SLAYER**](https://github.com/bamsumit/slayerPytorch). src/loss.py calculates loss functions.
- utils - used for loading pretrained models (rbase.py) and GPU management (gpu.py)
 
## Using the Code for Testing

### Setup
Setup your python virtual env using the requirements.txt document.
```
pip install -r requirements.txt
```
- Python3/3.7.0
- Ubuntu 18.04.4 LTS
- cuda/10.0.130 cudnn/v7.5.0
- Nvidia GPUs (Quadro P6000)

### Download Datasets
Download preformatted datasets here: https://drive.google.com/drive/folders/1yrHUqYf0rWrfxbQILzKB9_kDYWF6yekd
Must download entire folder, with the "img" subfolder and hdf5 file. 

For example, if processing EVIMO wallseq00, the datafolder should have the following structure:

```
wallseq00
- img
-- depth_mask_5.png
-- depth_mask_6.png 
-- ...
- wallseq00.hdf5
```

The format of the hdf5 is as follows:

- events (N x 4 - t, x, y, polarity)
- events_idx (nFrames x 1 - starting index of event for frame n)
- num frames (scalar - total frames available)
- timeframes (nFrames x 3 - [minFrameWindowTime, maxFrameWindowValidTime, depth_mask_N.png N])

Only events within +- X seconds of the time a mask was recorded are used. This time window (--valid_time) is set in preprocessing scripts provided (--valid_time=0.01 for EVIMO hdf5 files in the drive folder).

Some timeframes are very sparse and don't meet the minimum number of events per timeframe. Events from these timeframes are filtered out. This threshold (--min_events) is set in preprocessing scripts provided (--min_events=1000 for EVIMO hdf5 files in the drive folder).



### Run pretrained model

To test the SpikeMS code, you'll need to run test.py. The script contains the following command line args:

- `modeltype`: Name of Spiking Neural Network Model used. Modeltype specifies NN architecture. Valid modeltypes specified in models/utils.py
- `segDatasetType`: Type of dataset used. EVIMO, MOD, MVEC options available.

- `datafile`: Name of data file contain event and timeframe data. NOTE: must be in the hdf5 format generated through  using scripts included in preprocessingHelpers. Unique to each DatasetType.
- `maskDir`: Relative directory containg 2d mask images. EVIMO assumes naming convention "depth_mask_#.png"
- `logdir`: Logging directory. If it does not exist, will be generated.
- `config`: Path to config file. NOTE: dataset specific options, including 2D eventframe dimensions must be specified in config file.  Specfied under "data" config parameters. Descriptions of parameters included under general_config.yaml. 
- `checkpoint`: Path to checkpoint file. Used to load pretrained models.

- `saveImages`: Enable to save images produced during processing
- `saveImageInterval`: Save images periodically with this period/interval. I.e. If interval is 5, will save images every 5 iterations.
- `imageLabel`: optional suffix added to in-progress images 

- `crop`: Crop events in x,y dimentions to the height_c and width_c specified in the config file. Centered around densest point in event frame.
- `maxBackgroundRatio`: The max ratio of background events to object events. Used to filter input events. A larger allowed ratio decreases quality of model but utilizes more events.')
- `incrementalPercent`: Percentage of training time captured by each mask image used for testing time window.
   Filter input by ratio of background events to object events. A larger allowed ratio decreases quality of model but utilizes more events    

### Example: EVIMOWall and MOD
For paper results using a pretrained trained with EVIMOWall datasets and Cross Entropy Loss and SpikeLoss, run the following command (change maxBackgroundRatio to number in range of [1.5, 3] for best results):

```
#Test on EVIMO eval seq00 
python test.py  \
--crop --maxBackgroundRatio=1.5--segDatasetType="EVIMO" \
--modeltype="unetRNN6Layer_noBlock" \
--maskDir="/location/of/datafolder/eval_wall/seq_00/img" \
--datafile="/location/of/datafolder/eval_wall/seq_00/seq_00_v2.hdf5" \
--checkpoint="pretrainedModels/EVIMOWall-6Layer_crossEntropySpikeLoss/out/best_checkpoint.pth.tar" 

#Test on MOD room1obj1
python test.py  \
--crop --maxBackgroundRatio=1.5  --segDatasetType="MOD" \
--modeltype="unetRNN6Layer_noBlock" \
--maskDir="/location/of/datafolder/MOD_seq_room1_obj1/masks" \
--datafile="/location/of/datafolder/MOD_seq_room1_obj1/room1obj1.hdf5" \
--checkpoint="pretrainedModels/EVIMOWall-6Layer_crossEntropySpikeLoss/out/best_checkpoint.pth.tar"  

```
#### Incremental Prediction
Use the `--incrementalPercent` command line to limit the number of events used event frame. 

For example, let's set --incrementalPercent=0.5 using events processed with a time window of valid_time=0.01. If an original time window contained all events between t=0 and t=0.02, only events between t=0 and t=0.01 will be used.
 
#### Save Images 
Use these command line arguments to save 2D images of the processing and output results.
```
--saveImages
--saveImageInterval (default=1) 
--imageLabel (default="")
```
Saves images with interval saveImageInterval to subfolder `images` under logDir.
    

#### Config file
This uses by default the configuration file in `general_config.yaml`.
Modify the test config if you want to change one of the following parameters:
- Batch size
- Number of reader threads
- GPU device number

Dataset-specific variables including image dimensions and desired cropped image dimensions also specified in config.

## Acknowledgements
Several sections of the code are based on [**SLAYER**](https://github.com/bamsumit/slayerPytorch) and [**Event-Based Angular Velocity Regression with Spiking Networks**](https://github.com/uzh-rpg/snn_angular_velocity)
