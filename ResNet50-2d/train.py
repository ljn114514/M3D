import os, random, torch, dataset, cv2, time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import resnet

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
###########   HYPER   ###########
base_lr = 0.001
momentum = 0.9
weight_decay = 5e-4
gamma = 0.1

num_epoches = 20
step_size = 12
batch_size = 128
##########   DATASET   ###########
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([ transforms.ToTensor(),  normalizer, ])

img_dir = 'YourDataset/Mars/bbox_train/'
train_dataset = dataset.imgdataset(dataset_dir=img_dir, txt_path='list/list_train.txt', new_height=256, new_width=128, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

test_dataset = dataset.imgdataset(dataset_dir=img_dir, txt_path='list/list_val.txt', new_height=256, new_width=128, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=batch_size/4, shuffle=False, num_workers=4)
###########   MODEL   ###########
#resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth
model = resnet.resnet50(pretrained='resnet50-19c8e357.pth', num_classes=625, train=True)
model.cuda()
criterion = nn.CrossEntropyLoss()

param = []
params_dict = dict(model.named_parameters())
new_param = ['feat.weight', 'feat.bias', 'feat_bn.weight', 'feat_bn.bias', 'classifier.weight', 'classifier.bias']
for key, v in params_dict.items():	
	if key in new_param:
		param += [{ 'params':v,  'lr_mult':10}]
	else:
		param += [{ 'params':v,  'lr_mult':1}]
optimizer = torch.optim.SGD(param, lr=base_lr, momentum=momentum, weight_decay=weight_decay)

###########   TRAIN   ###########
def adjust_lr(epoch):
	lr = base_lr * (gamma ** (epoch // step_size))
	for g in optimizer.param_groups:
		g['lr'] = lr * g.get('lr_mult', 1)
	return lr

for epoch in range(num_epoches):
	lr = adjust_lr(epoch)
	print('-' * 10)
	print('epoch {}'.format(epoch + 1))

	running_loss = 0.0
	running_acc = 0.0
	start = time.time()
	since = time.time()
	
	model.train()
	for i, data in enumerate(train_loader, 1):
		images, label = data
		images = Variable(images).cuda()
		label = Variable(label).cuda()
		out = model(images)

		loss = criterion(out, label)
		#print 'running_loss: ', running_loss
		running_loss += loss.data[0] * label.size(0)
		_, pred = torch.max(out, 1)
		num_correct = (pred == label).sum()
		running_acc += num_correct.data[0]
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if i % 500 == 0:
			print('[{}/{}] iter: {}/{}. lr: {} . Loss: {:.6f}, Acc: {:.6f} time:{:.1f} s'.format(epoch+1, num_epoches, i, len(train_loader), lr, running_loss/(batch_size*i), running_acc/(batch_size*i), time.time() - since))
			since = time.time()
	print('[{}/{}] iter: {}/{}. lr: {} . Loss: {:.6f}, Acc: {:.6f}'.format(epoch+1, num_epoches, i, len(train_loader), lr, running_loss/(batch_size*i), running_acc/(batch_size*i)))
	print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch+1, running_loss/(len(train_dataset)), running_acc/(len(train_dataset))))

	model.eval()
	eval_loss = 0.
	eval_acc = 0.
	for data in test_loader:
		images, label = data
		images = Variable(images, volatile=True).cuda()
		label = Variable(label, volatile=True).cuda()
		out = model(images)

		loss = criterion(out, label)
		eval_loss += loss.data[0] * label.size(0)
		_, pred = torch.max(out, 1)
		num_correct = (pred == label).sum()
		eval_acc += num_correct.data[0]
	print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss/(len(test_dataset)), eval_acc/(len(test_dataset))))
	print('Time:{:.1f} s'.format(time.time() - start))

	#if (epoch+1)%2 == 0:
	torch.save(model.state_dict(), 'weight/resnet50_mars_%05d.pth'%(epoch))
