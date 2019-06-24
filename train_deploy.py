#Author Mark Abrenio / 26 May 2019

# Imports here
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import random

from torchvision import datasets
from torch.utils.data import DataLoader

from torch.autograd import Variable

from PIL import Image

#from predict import predict

data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets
#data_transforms = transforms.Compose([transforms.ToTensor(),
#                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                       ])

data_transforms = {
	'train': transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

def plot_categories(data):
	try:
		import numpy as np
		objects = list(data.keys())
		performance = list(data.values())
		y_pos = np.arange(len(objects))

		plt.bar(y_pos, performance, align='center', alpha=0.5)
		plt.xticks(y_pos, objects)
		plt.ylabel('PROBABILITY')
		plt.title('TOP 5 PROBABILITIES')

		plt_result = plt.show()
		print "plotted"

	except Exception as e:
		print e
		print "Cannot Plot Categories!"

def my_top_5(outputs):
	data_for_plot = {}
	sm = torch.nn.Softmax(dim = 1)
	pop = sm(outputs)
	da_valuez, predictedTop5_test = pop.topk(5)
	da_valuez = da_valuez.tolist()
	da_valuez = da_valuez[0]
	predictedTop5_test = predictedTop5_test.tolist()
	predictedTop5_test = predictedTop5_test[0]
	throttle = 0
	while throttle < 5:
		da_top = predictedTop5_test[throttle]
		da_probability = da_valuez[throttle]
		da_probability = da_probability * 100
		real_label = str(da_top).decode("utf-8")
		pred_flower_name = cat_to_name[real_label]
		print pred_flower_name + " Probability: " + str(da_probability)
		throttle += 1
		data_for_plot[pred_flower_name] = da_probability #lets build the dictionary for the matplotlib
	return data_for_plot #*return the dictionary for matplotlib


def predict(model, da_transform, image):
	image = Image.open(image)
	img = da_transform(image)
	img = img.unsqueeze(0)
	outputs = model(img)
	print "Great Success: Pushed image through model!"
	data = my_top_5(outputs)
	try:
		plot_categories(data)
	except Exception as e:
		print e
		print "ERROR: Could Not Plot Probabilities"

def listdir_fullpath(path):     #returns all_da_symbols!

	for_return = []

	for path, subdirs, files in os.walk(path):
		for name in files:
			if "DS_Store" not in name:
				if name.endswith(".jpg"):
					for_return.append(os.path.join(path, name))

	return for_return

def fine_tune_model(model):
	n_classes = 0
	for param in model.parameters():    #lets just set them to no grad
			param.requires_grad = False
	n_inputs = 4096
	n_classes = len(cat_to_name.keys())
	dataset_sizes["train"] = len(listdir_fullpath(train_dir)) #***total num of images, for loss func
	dataset_sizes["eval"] = len(listdir_fullpath(test_dir)) #***total num of images, for loss func
	# Add on classifier
	model.classifier[6] = nn.Sequential(
					  nn.Linear(n_inputs, 256),
					  nn.ReLU(),
					  nn.Dropout(0.4),
					  nn.Linear(256, n_classes),
					  nn.LogSoftmax(dim=1))
	return model

# TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(train_dir, data_transforms["train"])

print "All Images, for Training:  "
print image_datasets.classes

evaluate_dataset = datasets.ImageFolder(test_dir, data_transforms["val"])

print "Eval Images, For Test: "
print evaluate_dataset.classes

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {}

train_loader = DataLoader(image_datasets, batch_size=100 ,shuffle=True, num_workers=1)

eval_loader = DataLoader(evaluate_dataset, batch_size=1 ,shuffle=True, num_workers=1)

dataloaders["train"] = train_loader #***well break up train and eval into seperate functions

import json

with open('cat_to_name.json', 'r') as f:
	cat_to_name = json.load(f)

dataset_sizes = {}
dataset_sizes["train"] = len(image_datasets)


# TODO: Build and train your network

#***LOAD PRETRAINED

from torchvision import models
import torch.nn as nn

import sys		#***lets handle command line argz!
arg_count = len(sys.argv)
learning_rate = 0.001 #**this is the default learning rate!
learning_rate_set = False
manual_model_set_from_file = False
train_decision = False
evaluate_decision = False
gpu_use_decision = False
epoch_count_set = False
arc_choice = False

train_by_default = True

number_epochs = 0
man_model = ""
if arg_count > 1:
	throttle = 0
	for x in sys.argv:
		if x == "-l":
			learning_rate = sys.argv[throttle + 1]
			learning_rate_set = True
		if x == "-m":
			man_model = sys.argv[throttle + 1]
			manual_model_set_from_file = True
		if x == "-t":
			train_status = sys.argv[throttle + 1]
			if train_status == "y" or train_status == "yes":
				train_decision = True
				train_by_default = False
		if x == "-e":							#**CL arg to do evalutaion test of model. "-e yes"
			evaluate_status = sys.argv[throttle + 1]
			if evaluate_status == "y" or evaluate_status == "yes":
				evaluate_decision = True
		if x == "-gpu":
			gpu_status = sys.argv[throttle + 1]
			if gpu_status == "y" or gpu_status == "yes":
				gpu_use_decision = True
		if x == "-ec":
			number_epochs = sys.argv[throttle + 1]
			number_epochs = int(number_epochs)
			epoch_count_set = True
		if x == "-ac":
			chosen_arc = sys.argv[throttle + 1]
			arc_choice = True
			manual_model_set = True
		throttle += 1

manual_model_load = False
arch_in_use = 0

if manual_model_set_from_file == False: #**They want to
	if arc_choice:
		if chosen_arc == "vgg16":
			model = models.vgg16(pretrained=True)  #lets go ahead and use the VGG16 pretrained
			manual_model_load = False
			arch_in_use = "vgg16"
		elif chosen_arc == "alexnet":
			model = models.alexnet(pretrained=True)
			arch_in_use = "alexnet"
		else:
			print "You Did Specify An Available Model Type. Defaulting to VGG16"
			model = models.vgg16(pretrained=True)  # lets go ahead and use the VGG16 pretrained
			manual_model_load = False
			arch_in_use = "vgg16"
	else:
		print "ERROR: You Did not choose model type defaulting to AlexNet"
		model = models.alexnet(pretrained=True)  # lets go ahead and use the VGG16 pretrained
		manual_model_load = False
		arch_in_use = "alexnet"

if manual_model_set_from_file:
	model = torch.load(man_model, map_location='cpu')

print model

if manual_model_set_from_file == False:
	model = fine_tune_model(model)

print "Updated Model: "
print model

#***END LOAD PRETRAINED


#***BEGIN manual scan


def manual_scan_test(model, test_data_loader):
	model.eval()	#we do not need grads
	correct_accum = 0
	total_items = len(test_data_loader)

	for inputs, labels in test_data_loader:
	#if cuda_available:
		#                inputs = Variable(inputs.cuda())	#**lets put them onto the GPU
		#                labels = Variable(labels.cuda())
		try:
			outputs = model(inputs)
			_, prediction = torch.max(outputs.data, 1)
			correct_accum += torch.sum(prediction == labels.data)
			prediction = prediction.cpu().numpy()
			prediction = int(prediction)
			prediction = str(prediction).decode("utf-8")
			real_label = labels.cpu().numpy()
			real_label = int(real_label)
			real_label = str(real_label).decode("utf-8")
			pred_flower_name = cat_to_name[prediction]
			real_flower_name = cat_to_name[real_label]
			print "Flower is: " + real_flower_name
			print "Predicted Flower is: " + pred_flower_name
			print "-----------"
		except Exception as e:
			print e
			print "Couldnt Predict This one."

	print "Total Items: " + str(total_items)
	print "Correctly Predicted: " + str(correct_accum)

#***END manual scan

#***BEGIN TRAIN MODEL
cuda_available = torch.cuda.is_available()
if cuda_available:
	print "SUCCESS: We have CUDA"
else:
	print "ERROR: NO GPU AVAILABLE"

#if cuda_available:
#	model.cuda() #**its too slow so we need to push it onto the GPU!! 
#	print "Model ON GPU"

def model_checkpoint_load(da_model_file, architecture, gpu_use):
	print "Loading Model State From File"
	try:
		checkpoint = torch.load(da_model_file)
		if "alex" in architecture:
			#model = getattr(torchvision.models, models.alexnet(pretrained=True))
			model = models.alexnet(pretrained=True)
		if "vgg" in architecture:
			#model = getattr(torchvision.models, models.vgg16(pretrained=True))
			model = models.vgg16(pretrained=True)
		for parameter in model.parameters(): #***freeze current params we place classifier below
			parameter.requires_grad = False

		model.classifier = checkpoint['classifier']
		model.load_state_dict(checkpoint['state_dict'])
		model.optimizer = checkpoint['optimizer']
		model.eval()
		if gpu_use:
			print "pushing model to GPU"
			model.cuda()
		return model
	except Exception as e:
		print e
		print "Could Not Load Model Check Point I sorry"
		return 0

def save_models(epoch, model): #*only save state_dict, less file size
	print "Saving Model State To File"
	try:
		file_ext = str(random.randint(1, 99999))
		file_name = "LIVE_CNN_model_Statc_Dict.model." + file_ext
		checkpoint = {'input_size': (3, 224, 224),
					  'output_size': 102,
					  'batch_size': 100,
					  'learning_rate': learning_rate,
					  'model_name': "AIPND2",
					  'state_dict': model.state_dict(),
					  'optimizer': optimizer.state_dict(),
					  'epoch': epoch,
					  'classifier': model.classifier}
		torch.save(checkpoint, file_name)
		return file_name #***RETURN FILE NAME OF SAVED MODEL!
	except Exception as e:
		print e
		print "Could not save model to disk!"
		return 0

def reload_model(epoch, model):
	try:
		model_fn = save_models(epoch, model)
		model = torch.load(model_fn)
		return model
	except Exception as e:
		print e
		return 0

def train_model(model, criterion, optimizer, scheduler, architecture, gpu_use, num_epochs=101):
	since = time.time()

	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train']:
			if phase == 'train':
				#scheduler.step()
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for inputs, labels in dataloaders[phase]:
				#print "iter"
				#inputs = inputs.to(device)
				#labels = labels.to(device)

				if cuda_available and gpu_use:
					inputs = Variable(inputs.cuda())
					labels = Variable(labels.cuda())

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]
		if epoch % 10 == 0: #**Lets save ever 10 epochs!
				print "Saving and Reloading Model"
				da_model_file = save_models(epoch, model)
				model_checkpoint_load(da_model_file, architecture, gpu_use)
		print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
		print "Total Images: " + str(dataset_sizes[phase])
		print "Total Correctly Predicted: " + str(int(running_corrects))


#***END TRAIN MODEL

from torch.optim import Adam
import torch.optim as optim

#optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.9)

scheduler = 0

evaluate_existing_model = True

num_epochs = 61 #***default num of epochs

if epoch_count_set:
	num_epochs = number_epochs
	print "NUMBER OF EPOCHS: " + str(num_epochs)
else:
	print "Using Default Epoch Cound: 101"

if train_decision or train_by_default: #**will use GPU by default for training unless they specify no
	if cuda_available and gpu_use_decision:
			model.cuda() #**its too slow so we need to push it onto the GPU!!
			print "Model ON GPU"
			gpu_use = True
			train_model(model, criterion, optimizer, scheduler, arch_in_use, gpu_use, num_epochs)
	else:
		print "NOT USING GPU. Going to be Slow."
		gpu_use = False
		train_model(model, criterion, optimizer, scheduler, arch_in_use, gpu_use, num_epochs)
else:
	print "Decided not to train"

print "Testing Prediction Function"
predict(model, data_transforms['val'], "test_image.jpg")

if evaluate_decision:
	manual_scan_test(model, eval_loader)
