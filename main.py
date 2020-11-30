# By Ali Hatamizadeh 
import torch
from torchvision import models, datasets, transforms
from torch import nn
import glob
import numpy as np
import PIL
import cv2
from matplotlib import pyplot as plt
import argparse

class Network_Hook(nn.Module):
    """
    Submodule for registering hooks on target layers in the trained network
    """
    def __init__(self, model, target_layers):
        super().__init__()
        self.model = model
        self._features = {layer: torch.empty(0) for layer in target_layers}
        for layer_id in target_layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        """
        Args:
            layer_id: name of a layer in the network
        Returns:
            Extracted feature for the input layer
        """
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        """
        Args:
            x: (torch.Size([B, 3, H, W]))
        Returns:
            Dictionary of extracted features from the target layers
        """
        _ = self.model(x)
        return self._features

class Visual_Computer(nn.Module):
    """
    Model-agnostic module for computing CAMs, given target layers and trained model.
    It registers hooks on the network layers and computes the class activation maps based on extracted features and weights corresponding to the last classification layer for the desired class.
    """
    def __init__(self, model, target_layers):
        super().__init__()
        self.target_layers = target_layers
        self.model = model
        self.Registered_Hooks = Network_Hook(self.model, target_layers=self.target_layers)

    def find_index(self,logit):
        """
        Args:
            logit: logits computed from the last classification layer
        Returns:
            index of the class

        """
        _, index = torch.nn.functional.softmax(logit, dim=1).data.squeeze().sort(0, True)

        return index.cpu().numpy()[0]



    def normalize_probmap(self,cam):
        """
        Args:
            cam: un-normalized Class activation maps: numpy.ndarray(size=(H, W), dtype=np.float)
        Returns:
            Normalized probability values for CAM: numpy.ndarray(size=(H, W), dtype=np.float)

        """
        cam -= cam.min()
        cam /= cam.max()

        return 1-cam

    def compute_CAM(self,feature_extract, weights, index ,img_size):
        """
        Args:
            feature_extract: extracted features from target layer: numpy.ndarray(size=(1,c,h, w), dtype=np.float) where c,h,w are the channel dimention and spatial sizes of the layer output (e.g. 512,7,7)
            weights: shape of the weights: numpy.ndarray(size=(d, c), dtype=np.float)
            index: index of the class
            img_size: shape of the image (H, W)
        Returns:
           Computed class activation maps: (np.ndarray(dtype=np.float32, size=(H, W)))

        """
        _, ch, H, W = feature_extract.shape
        cam = weights[index].dot(feature_extract.reshape((ch, H*W))).reshape(H, W)
        cam = self.normalize_probmap(cv2.resize(cam, img_size, interpolation=cv2.INTER_CUBIC))

        return np.uint8(255 * cam)

    def forward(self, x):
        """
        Args:
            x: (torch.Size([B, 3, H, W]))
        Returns:
            Computed class activation maps: (np.ndarray(dtype=np.float32, size=(H, W)))
        """
        hooked_features = self.Registered_Hooks(x)
        extracted_features = hooked_features[self.target_layers[0]].detach().cpu().numpy()
        index = self.find_index(hooked_features[self.target_layers[1]])
        weight = model.fc.weight.detach().cpu().numpy()
        cam = self.compute_CAM(extracted_features, weight, index, (w, h))
        return cam

def img_preprocess(Image,device):
    """
    Args:
        Image: Input image (np.ndarray(dtype=np.float32, size=(H, W)))
    Returns:
        Preprocessed tensor of the input image placed on device (torch.Size([B, 3, H, W]))
    """

    Image = preprocess(Image)
    Image = Image.unsqueeze(0).to(device)

    return Image

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--target_layer', default='layer4', type=str,help='Specify the name of the target layer (before the pooling layer)')
    parser.add_argument('--final_layer', default='fc', type=str,help='Specify the name of the last classification layer')
    args = parser.parse_args()
    device = torch.device('cuda:'+args.device)
    model = torch.load('resnet-cam.pt', map_location=device)   # Demo for a ResNet-34 trained on STL-10 Dataset (https://cs.stanford.edu/~acoates/stl10/)
    print(model)
    model.cuda()
    model.eval()
    cam_computer = Visual_Computer(model, target_layers=[args.target_layer,args.final_layer])
    resize_param = (224, 224)
    norm_mean = [0.5528, 0.5528, 0.5528]
    norm_std = [0.1583, 0.1583, 0.1583]
    disp_size = 10
    preprocess = transforms.Compose([transforms.Resize(resize_param),transforms.ToTensor(),transforms.Normalize(norm_mean, norm_std)])
    plt.figure(figsize=(disp_size, disp_size))
    for i, file in enumerate(glob.glob('./test_images/*')):
        image = PIL.Image.open(file)
        h, w, b = np.shape(np.array(image))
        img_tensor = img_preprocess(image,device)
        cam_img = cam_computer(img_tensor)
        img = np.array(image)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.6
        plt.subplot(2, 1, i + 1)
        plt.imshow(result.astype(np.int))

    plt.show()




