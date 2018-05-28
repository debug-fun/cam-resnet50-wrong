# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
from config import *
import io
import os
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import MyDensenet as desenet
import MyResNet as resent
import torch
import torch.optim as optim
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# input image
LABELS_URL = 'labels.json'
#IMG_URL = '9933031-large.jpg'
#IMG_URL = '/home/wd4t/hx/download_test/totalTest/test/ffffd526f7e18b49.jpg'
IMG_URL = '/home/hx/googlelandmark/download1000/7172/fe34c7cd848e7ef9.jpg'
# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
# model_id = 2
# if model_id == 1:
#     net = models.squeezenet1_1(pretrained=True)
#     finalconv_name = 'features' # this is the last conv layer of the network
# elif model_id == 2:
#     net = models.resnet18(pretrained=True)
#     finalconv_name = 'layer4'
# elif model_id == 3:
#     net = models.densenet161(pretrained=True)
#     finalconv_name = 'features'

#img = Image.open(" ")
#net = desenet.densenet201(False, num_classes=14951)
net=resent.resnet50(False,num_classes=num_classes)
if USE_CUDA:
    net = nn.DataParallel(net).cuda()
net.load_state_dict(torch.load("/home/wd4t/weight/weightnodrop1_14_total_resnet521"))
#net.load_state_dict(torch.load("/home/wd4t/hx/weight/weight_3"))
net.eval()

#vlad, c = net(img)

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
#
# net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img_pil = Image.open(IMG_URL)
img_pil.save('test.jpg')

img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
#logit = net(img_variable)
vlad, logit = net(img_variable)

# download the imagenet category list
# classes = {int(key):value for (key, value)
#           in json.load(open(LABELS_URL)).items()}
classes = range(14951)

h_x = F.softmax(logit).data.squeeze()
probs, idx = h_x.sort(0, True)

# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction
#CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
CAMs = returnCAM(vlad.cpu().data.numpy(), weight_softmax, [idx[0]])

# render the CAM and output
print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
img = cv2.imread('test.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap*0.8+img*0.5
#result=heatmap
#cv2.imshow("img",result)
#cv2.waitKey(0)
cv2.imwrite('CAM.jpg', result)
