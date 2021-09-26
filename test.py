import argparse
import os
# Must be set before importing torch.
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from config.utils import getTestConfigs
from runner import Runner

def main():
    parser = argparse.ArgumentParser(description='Test the SNN model')

    parser.add_argument('--modeltype',
                        help='model type')

    parser.add_argument('--segDatasetType',
                        default="EVIMO",
                        help='Only EVIMO datasets available.')

    parser.add_argument('--datafile',
                        help='Name of data file (in hdf5 format)')
    
    parser.add_argument('--maskDir',
                        help='directory containg 2d mask images. EVIMO assumes naming convention "depth_mask_#.png"')

    parser.add_argument('--checkpoint',
                        default='/',
                        help='checkpoint file name')

    parser.add_argument('--logdir',
                        default=os.path.join(os.getcwd(), 'logs'),
                        help='Test logging directory')

    parser.add_argument('--config',
                        default=os.path.join(os.getcwd(), 'general_config.yaml'),
                        help='Path to test config file')

    #processing options
    parser.add_argument('--crop', help='crop to the height_c and width_c specified in the config file', dest='feature', action='store_true')
    parser.set_defaults(crop=False)

    parser.add_argument('--maxBackgroundRatio',
                        default=2, type=float,
                        help='Filter input by ratio of background events to object events. A larger allowed ratio decreases quality of model but utilizes more events.')
    
    parser.add_argument('--incrementalPercent',
                        default=1, type=float,
                        help='percentage of training time used captured by each mask image used for testing time window, for Incremental Prediction')
    
    #image options

    parser.add_argument('--saveImages',
                        default=False,
                        help='Save images produced during processing')

    parser.add_argument('--saveImageInterval',
                        default=1, type=float,
                        help='Save images periodically with this period/interval')

    parser.add_argument('--imageLabel',
                        default="",
                        help='optional suffix added to in-progress images ')



    
    args = parser.parse_args()
    
    #sets up logging configs and dirs
    configs = getTestConfigs(args.logdir, args.config)

    if(args.saveImages):
        imageDir = os.path.join(configs['log'].getOutDir(), "images")
    else:
        imageDir = None

    runner = Runner(args.crop, args.maxBackgroundRatio, args.segDatasetType, args.datafile,
           args.checkpoint, args.modeltype,
            configs['log'], args.config,
            args.maskDir, args.incrementalPercent,
            args.saveImages, args.saveImageInterval, imageDir, args.imageLabel)

    print("\nstarting to run {} {}".format(args.segDatasetType, args.modeltype))
    runner.test()

if __name__ == '__main__':
    main()
