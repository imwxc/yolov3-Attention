from models import *
import torch
from senet.se_resnet import *
from senet.se_inception import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_def(opt):
    if opt.model_def == "config/yolov3.cfg":
        return Darknet(opt.model_def, img_size=opt.img_size).to(device)
    elif opt.model_def == "config/yolov3-tiny.cfg":
        return Darknet(opt.model_def, img_size=opt.img_size).to(device)
    elif opt.model_def=="SE-resnet18":
        return se_resnet18()
    elif opt.model_def=="SE-resnet34":
        return se_resnet34()
    elif opt.model_def=="SE-resnet50":
        return se_resnet50()
    elif opt.model_def=="SE-resnet101":
        return se_resnet101()
    elif opt.model_def=="SE-resnet152":
        return se_resnet152()
    elif opt.model_def == "SE-inceptionV3":
        return se_inception_v3(num_classes=1000)

def load_pretrain(model,opt):
    if("yolo" in opt.model_def):
        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(opt.weights_path))
    elif("SE" in opt.model_def):
        model.load_state_dict(torch.load("./checkpoints/"))
    return model
