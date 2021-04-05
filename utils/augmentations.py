import imgaug.augmenters as iaa
from .transforms import *

class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.2)),
            # iaa.Affine(rotate=(-20, 20)),#,translate_percent=(-0.05,0.05)),  # rotate by -45 to 45 degrees (affects segmaps) translate_percent(-0.2,0.2)会导致target出错
            iaa.AddToBrightness((-30, 30)), 
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.2),
            iaa.Grayscale(alpha=(0.0, 0.05))
        ], random_order=True)


AUGMENTATION_TRANSFORMS = transforms.Compose([
        AbsoluteLabels(),
        DefaultAug(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ])
