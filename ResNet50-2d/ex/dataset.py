import os, torch, random, cv2
import numpy as np
from torch.utils import data

class imgdataset(data.Dataset):
	def __init__(self, dataset_dir, txt_path, new_height, new_width, transform):
		self.new_height = new_height
		self.new_width = new_width
		self.dataset_dir = dataset_dir
		self.transform = transform

		with open(txt_path) as f:
			line = f.readlines()
			self.img_list = [os.path.join(dataset_dir, i.split()[0]) for i in line]
			self.label_list = [int(i.split()[1]) for i in line]

	def __getitem__(self, index):
		im_path = self.img_list[index]		  
		if im_path.split()[-1] in ['png', 'jpg', 'jpeg']:
			return
		image = cv2.imread(im_path)
		image = cv2.resize(image,(self.new_width, self.new_height))
		image = image[:,:, ::-1]
		#print np.shape(image)
		#image = np.array(image, np.float32)
		#image = np.transpose(image, (2, 0, 1))
		#image = torch.from_numpy(image).float()
		image = self.transform(image.copy())
		return image, self.label_list[index]

	def __len__(self):
		return len(self.label_list)