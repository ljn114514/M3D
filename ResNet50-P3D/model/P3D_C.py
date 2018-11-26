import torch.nn as nn
import math, torch
import torch.utils.model_zoo as model_zoo
from torch.nn import init

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()

		self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm3d(planes)

		self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
		self.bn2 = nn.BatchNorm3d(planes)
		self.avgpool_s = nn.AvgPool3d(kernel_size = (stride,1,1), stride=(stride, 1, 1), padding=0)

		self.conv2_t = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(1, 0, 0), bias=False)
		self.bn2_t = nn.BatchNorm3d(planes)
		self.conv2_t.weight.data.fill_(0)
		
		self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm3d(planes * 4)

		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out_t = self.conv2_t(out)
		out_t = self.bn2_t(out_t)
		out_t = self.relu(out_t)

		out = self.avgpool_s(out)
		out = out + out_t

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=1000, train=True):
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.istrain = train

		self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
		self.bn1 = nn.BatchNorm3d(64)		

		self.conv1_t = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
		self.bn1_t = nn.BatchNorm3d(64)
		self.conv1_t.weight.data.fill_(0)

		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool3d((1,8,4), stride=1)

		self.num_features = 128
		self.feat = nn.Linear(512 * block.expansion, self.num_features)
		self.feat_bn = nn.BatchNorm1d(self.num_features)
		init.kaiming_normal(self.feat.weight, mode='fan_out')
		init.constant(self.feat.bias, 0)
		init.constant(self.feat_bn.weight, 1)
		init.constant(self.feat_bn.bias, 0)
		self.drop = nn.Dropout(0.5)
		self.classifier = nn.Linear(self.num_features, num_classes)
		init.normal(self.classifier.weight, std=0.001)
	 	init.constant(self.classifier.bias, 0)

		for m in self.modules():
			if isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv3d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm3d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x_t = self.conv1_t(x)
		x_t = self.bn1_t(x_t)
		x_t = self.relu(x_t)

		x = x + x_t
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.feat(x)

		if self.istrain:
			x = self.feat_bn(x)
			x = self.relu(x)
			x = self.drop(x)
			x = self.classifier(x)
		return x

def resnet50(pretrained='True', num_classes=1000, train=True):
	model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, train)

	weight = torch.load(pretrained)
	static = model.state_dict()
	for name, param in weight.items():
		if name not in static:
			print 'not load weight ', name, param.size(), param.dim()
			continue

		if isinstance(param, nn.Parameter):
			param = param.data
			
		if param.dim()>2:
			print('load weight 3d conv: ', name, static[name].size())
			param = param.unsqueeze(2)
			try:
				static[name].copy_(param)
			except :
				print('error********************')
				print(static[name].size(), param.size())
			
		else:
			print 'load weight: ', name, static[name].size()
			static[name].copy_(param)
	return model