import os, random, torch, dataset, cv2, time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import resnet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

##########   DATASET   ###########
batch_size = 8

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([ transforms.ToTensor(),  normalizer, ])

img_dir = '/media/disk2/ljn/video_dataset/Mars/bbox_test/'
test_dataset = dataset.videodataset(dataset_dir=img_dir, txt_path='list/list_test_seq_all.txt', new_height=256, new_width=128, 
	frames=8, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
###########   MODEL   ###########


pretrained = 'resnet50_3d_mars_00240.pth'
model = resnet.resnet50(pretrained=pretrained, num_classes=625, train=False)
model.cuda()
model.eval()

name = 'fea/fea'
output = open(name,'w')
num=0
for data in test_loader:
	
	num = num + batch_size
	images, label = data
	images = torch.transpose(images, 1, 2).contiguous()
	images = images.view(images.size(0)*images.size(1), images.size(2), images.size(3), images.size(4))
	images = Variable(images).cuda()

	out = model(images)
	fea = out.cpu().data
	fea = fea.numpy()

	for j in range(0, np.shape(fea)[0]):
		str1 = ''
		for x in range(len(fea[j])):
			str1 = str1 + str(fea[j][x]) +' '
		str1 = str1 + '\n'
		output.write(str1)
output.close()