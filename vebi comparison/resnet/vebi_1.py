from torchvision import transforms
import torchvision.models as models
import numpy as np
import os
import pickle
from PIL import Image
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from tqdm import tqdm

# Pretrained model
model = models.resnet50(pretrained=True)
model.eval()

print(model)

return_layers_all = {
    'layer1.0': 'cnn_1',
    'layer1.1': 'cnn_2',
    'layer1.2': 'cnn_3',
    'layer2.0': 'cnn_4',
    'layer2.1': 'cnn_5',
    'layer2.2': 'cnn_6',
    'layer2.3': 'cnn_7',
    'layer3.0': 'cnn_8',
    'layer3.1': 'cnn_9',
    'layer3.2': 'cnn_10',
    'layer3.3': 'cnn_11',
    'layer3.4': 'cnn_12',
    'layer3.5': 'cnn_13',
    'layer4.0': 'cnn_14',
    'layer4.1': 'cnn_15',
    'layer4.2': 'cnn_18'
}

t = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

t2 = transforms.Compose([
    transforms.Resize(224)
])

# values to normalize input
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean, std)

mid_getter = MidGetter(model, return_layers=return_layers_all, keep_output=True)


def binary(label):
    l = [0] * 1000
    l[label] = 1
    return l


def extract(mid):
    tmp = dict()
    for key in mid:
        tmp[key] = np.linalg.norm(mid[key].detach().numpy(), axis=(2, 3)).squeeze()
    #mid = np.concatenate(tmp)
    #del tmp
    return tmp


### Construct matrices

for label in tqdm(range(1000)):
    # skip if already processed
    if os.path.exists(f"pickle/{label}_X.pickle"):
        continue
    L = []
    X = []
    _dir = f"../data/val_set/{label}/"
    for filename in os.listdir(_dir + "img/"):
        # Read file
        inp = t(Image.open(_dir + "img/" + filename).convert('RGB'))[None, :, :]

        inp = normalize(inp)
        mid_outputs, _ = mid_getter(inp)

        L.append(binary(label))
        X.append(extract(mid=mid_outputs))

    with open(f"pickle/{label}_X.pickle", "wb") as f:
        pickle.dump(X, f)


