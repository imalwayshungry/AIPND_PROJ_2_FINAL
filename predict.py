#***Author: Mark Abrenio (2019)

# Imports here
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')


from torchvision import datasets
from torch.utils.data import DataLoader

from torch.autograd import Variable

import sys
import json

main_class_to_idx = 0

data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
real_test = data_dir + '/real_test'

def class_to_idx(prediction): #we need to use this to map the image class IDX to its name
    for key in main_class_to_idx:
        if main_class_to_idx[key] == prediction:
            prediction = int(key)
            return prediction  #either we return the right image class or unknown!
    return "unknown"

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

        plt.show()

    except Exception as e:
        print e
        print "Cannot Plot Categories!"

def my_top_5(outputs, k_count):
    data_for_plot = {}
    sm = torch.nn.Softmax(dim = 1)
    pop = sm(outputs)
    da_valuez, predictedTop5_test = pop.topk(k_count)
    da_valuez = da_valuez.tolist()
    da_valuez = da_valuez[0]
    predictedTop5_test = predictedTop5_test.tolist()
    predictedTop5_test = predictedTop5_test[0]
    throttle = 0
    for x in predictedTop5_test:
        real_class = class_to_idx(x)
        predictedTop5_test[throttle] = real_class
        throttle += 1
    throttle = 0
    while throttle < k_count:
        da_top = predictedTop5_test[throttle]
        da_probability = da_valuez[throttle]
        da_probability = da_probability * 100
        real_label = str(da_top).decode("utf-8")
        pred_flower_name = cat_to_name[real_label]
        print pred_flower_name + " Probability: " + str(da_probability)
        throttle += 1
        data_for_plot[pred_flower_name] = da_probability #lets build the dictionary for the matplotlib
    return data_for_plot #*return the dictionary for matplotlib

def listdir_fullpath(path):     #returns all_da_symbols!
    import os
    for_return = []

    for path, subdirs, files in os.walk(path):
        for name in files:
            if "DS_Store" not in name:
                if name.endswith(".jpg"):
                    for_return.append(os.path.join(path, name))

    return for_return


def predict(model, da_transform, image, gpu_use, k_count):
    model.eval()
    import PIL
    print image
    if k_count == 0:
        print "ERROR: k_count to low terminating"
        sys.exit(1)
    #image = Image.open(image)
    image = PIL.Image.open(image)
    #a = transforms.Resize(size=256, interpolation=PIL.Image.BILINEAR)
    #b = transforms.CenterCrop(size=(224, 224))
    #c = transforms.ToTensor()
    #img = a(image)
    #img = b(img)
    #img = c(img)
    img = da_transform(image)
    img = img.unsqueeze(0)
    if gpu_use:
        img = Variable(img.cuda())
    outputs = model(img)
    _, prediction = torch.max(outputs.data, 1)
    prediction = class_to_idx(int(prediction))
    print "Image Class Number Predicted:" + str(prediction)
    pred_flower_name = cat_to_name[str(prediction)]
    print "Predicted Flower Type: " + pred_flower_name
    try:
        data = my_top_5(outputs, k_count)
        plot_categories(data)
    except Exception as e:
        print e
        print "ERROR: Could Not Plot Probabilities"

def manual_scan_test(model, test_data_loader, gpu_use, k_count):
    if k_count == 0:
        print "ERROR: k_count to low terminating"
        sys.exit(1)
    model.eval()	#we do not need grads
    correct_accum = 0
    total_items = len(test_data_loader)

    for inputs, labels in test_data_loader:
    #if cuda_available:
        #                inputs = Variable(inputs.cuda())	#**lets put them onto the GPU
        #                labels = Variable(labels.cuda())
        try:
            if gpu_use:
                    inputs = Variable(inputs.cuda())	#**lets put them onto the GPU
                    labels = Variable(labels.cuda())
            outputs = model(inputs)
            _, prediction = torch.max(outputs.data, 1)
            correct_accum += torch.sum(prediction == labels.data)
            prediction = prediction.cpu().numpy()
            prediction = int(prediction)
            prediction = class_to_idx(prediction)
            prediction = str(prediction).decode("utf-8")
            real_label = labels.cpu().numpy()
            real_label = int(real_label)
            real_label = class_to_idx(real_label)
            real_label = str(real_label).decode("utf-8")
            pred_flower_name = cat_to_name[prediction]
            real_flower_name = cat_to_name[real_label]
            print "-----"
            print "Flower is: " + real_flower_name
            print "Predicted Flower is: " + pred_flower_name
            print "label: " + real_label
            my_top_5(outputs, k_count)
            print "-----\n"
        except Exception as e:
            print e
            print "Couldnt Predict This one."

    print "Total Items: " + str(total_items)
    print "Correctly Predicted: " + str(correct_accum)


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


if __name__ == "__main__":

    model = torch.load("saved_image_model1558125824.66", map_location='cpu')

    args_or_not = False
    arg_file = False
    arg_json = False
    gpu_decide = False
    model_file_specified = False
    prob_counts = 0
    json_name = ""
    if len(sys.argv) > 1:
        args_or_not = True

    throttle = 0

    #***EXAMPLE use for reviewer: predict.py -f test_image.jpg -k 3 -j cat_to_name.json

    for x in sys.argv:
        if x == "-f":   #script.py -f filename.jpg
            file_name = sys.argv[throttle + 1]
            arg_file = True
        if x == "-j":   #scrip.py -j json_file_name.json
            json_name = sys.argv[throttle + 1]
            arg_json = True
        if x == "-gpu":
            gpu_arg = sys.argv[throttle + 1]
            if gpu_arg == "yes" or gpu_arg == "y":
                gpu_decide = True
        if x == "-model":
            model_file = sys.argv[throttle + 1]
            model_file_specified = True
        if x == "-k":
            k_count = sys.argv[throttle + 1]
            prob_counts = int(k_count)
        throttle += 1

    torch.no_grad()

    print model

    if gpu_decide:
        print "You Chose To put model on GPU. Putting model on GPU"
        try:
            model.cuda()
        except Exception as e:
            print e
            print "ERROR: Could not put model on GPU. "
    else:
        print "Model Not put on GPU per your decision!"

    if arg_json:
        try:
            with open(json_name, 'r') as f:
                cat_to_name = json.load(f)
        except Exception as e:
            print e
            print "ERROR: Could not load the JSON file with the image classes. Did you specify a JSON file at the CL?"
            print sys.exit(1)

    evaluate_dataset = datasets.ImageFolder(test_dir, data_transforms["val"])
    print "Image Classes to IDX: "
    print evaluate_dataset.class_to_idx
    main_class_to_idx = evaluate_dataset.class_to_idx

    print "Image Classes to IDX: "

    if arg_file:
        predict(model, data_transforms['val'], file_name, gpu_decide, prob_counts)

    eval_loader = DataLoader(evaluate_dataset, batch_size=1 ,shuffle=True, num_workers=1)

    manual_scan_test(model, eval_loader, gpu_decide, prob_counts)