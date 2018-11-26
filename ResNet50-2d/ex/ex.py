import os, random, torch, dataset, cv2, time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import resnet
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

##########   DATASET   ###########
batch_size = 1

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([ transforms.ToTensor(),  normalizer, ])

img_dir = '../Mars/bbox_test/'
test_dataset = dataset.imgdataset(dataset_dir=img_dir, txt_path='list/list_mars_test.txt', new_height=256, new_width=128, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
###########   MODEL   ###########

for i in range(0,1):
	pretrained = '../weight/resnet50_mars_%05d.pth'%(i*2)
	model = resnet.resnet50(pretrained=pretrained, num_classes=625, train=False)
	model.cuda()
	model.eval()

	name = 'fea_test/fea%d'%(i)
	output = open(name,'w')
	num=0
	start_time = time.time()
	for data in test_loader:
		num = num+batch_size
		images, label = data
		images = Variable(images, volatile=True).cuda()
		label = Variable(label, volatile=True).cuda()
		out = model(images)
		fea = out.cpu().data
		fea = fea.numpy()

		print i, num, np.shape(fea)

		for j in range(0, np.shape(fea)[0]):
			str1 = ''
			for x in range(len(fea[j])):
				str1 = str1 + str(fea[j][x]) +' '
			str1 = str1 + '\n'
			output.write(str1)
	output.close()
	fea_ex_time = time.time() - start_time
	print fea_ex_time