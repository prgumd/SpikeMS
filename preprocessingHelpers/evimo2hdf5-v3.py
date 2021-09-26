import h5py
import numpy as np
import argparse
from PIL import Image
import os
from tqdm import tqdm
import shutil
import pathlib

files = os.listdir()
folders = [x for x in files if os.path.isdir(x)]

parser = argparse.ArgumentParser(description='Convert EVIMO txt files to hdf5 file')

parser.add_argument('-w', '--window_size',
                    default=0.01,
                    help='Only events within +- window_size of the time a mask was recorded are used. In seconds.')
parser.add_argument('-f','--folders', default=folders, nargs='+')
parser.add_argument('-m','--min_events', default=1000, help="min events per mask used")

args = parser.parse_args()

folders = args.folders
valid_time = float(args.window_size)
min_events = float(args.min_events)

for j, folder in enumerate(folders):
	print("Starting folder", folder)

	mask_imgs = []

	events = []
	masked_events =[]

	stevent_idx = []
	timeframes = []

	curr_mask_num = 0
	num_events = 0
	num_masked_events = 0

	reached_start = False
	rename_images = True

	# t, ex, ey, polarity
	data = np.loadtxt(os.path.join(folder,"events.txt"))
	dataset_txt = eval(open(os.path.join(folder,'meta.txt')).read())
	frames = dataset_txt['frames']

	curr_mask_time = frames[curr_mask_num]['cam']['ts']
	maskfilenm = os.path.join(folder, os.path.join('img', frames[curr_mask_num]['gt_frame']))
	mask = np.sum(np.asarray(Image.open(maskfilenm)),2)

	timeframes.append([curr_mask_time-valid_time,
		curr_mask_time+valid_time, 
		frames[curr_mask_num]['id']])
	toggle_newframe = True

	for i, event in enumerate(tqdm(data)):
		# print(event[0], "[", curr_mask_time - valid_time, curr_mask_time + valid_time , "] : ", curr_mask_num, "diff: ", curr_mask_time - curr_mask_num, np.sum(mask))
		#get to start
		if (not reached_start):
			if (event[0] >= curr_mask_time - valid_time):
				# print("reached", event[0], curr_mask_time)
				reached_start = True

		if (reached_start):
			#if event is outside mask window		
			if ((event[0] >= curr_mask_time + valid_time)):
				# print("outside ", event[0], ">", curr_mask_time + valid_time)
				if (curr_mask_num < len(frames) - 1):

					curr_mask_num += 1

					curr_maskfilenm = os.path.join(folder, os.path.join('img', frames[curr_mask_num]['gt_frame']))
					mask = np.asarray(Image.open(curr_maskfilenm))[:,:,0]

					while(np.sum(mask) < min_events and curr_mask_num + 1 < len(frames)):
						curr_mask_num += 1			
						curr_maskfilenm = os.path.join(folder, os.path.join('img', frames[curr_mask_num]['gt_frame']))
						try:
							mask = np.asarray(Image.open(curr_maskfilenm))[:,:,0]
						except:
							input(curr_maskfilenm)
							hf = h5py.File('{}.hdf5'.format(folder), 'w')

							hf.create_dataset('events',data=events)
							hf.create_dataset('events_idx',data=stevent_idx)
							hf.create_dataset('timeframes',data=timeframes)
							hf.create_dataset('numframes',data=len(timeframes))

							print("Created file with errors", folder)

							hf.close()
							break;	

					curr_mask_time = frames[curr_mask_num]['cam']['ts']
					timeframes.append([curr_mask_time-valid_time, curr_mask_time+valid_time, frames[curr_mask_num]['id']])
					print("update frame", curr_mask_num, curr_mask_time)
					toggle_newframe = True
				else:
					break

			if (event[0] >= curr_mask_time - valid_time  and
				event[0] <= curr_mask_time + valid_time):
				if (toggle_newframe):
					stevent_idx.append(num_events)
					toggle_newframe = False
				events.append(event)
				num_events += 1

	stevent_idx.append(max(num_events-1,0))

	hf = h5py.File('{}.hdf5'.format(folder), 'w')

	hf.create_dataset('events',data=events)
	hf.create_dataset('events_idx',data=stevent_idx)
	hf.create_dataset('timeframes',data=timeframes)
	hf.create_dataset('numframes',data=len(timeframes))

	print("Finished folder", folder)

	hf.close()



