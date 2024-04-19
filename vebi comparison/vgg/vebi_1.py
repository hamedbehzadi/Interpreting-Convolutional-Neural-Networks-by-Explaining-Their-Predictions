from torchvision import transforms
import torchvision.models as models
import numpy as np
import os
import pickle
from PIL import Image
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
#import spams
from tqdm import tqdm

# Pretrained model
model = models.vgg16(pretrained=True)
model.eval()

print(model)

return_layers_all = {
    'features.0': 'cnn_1',
    'features.2': 'cnn_2',
    'features.4': 'cnn_3',
    'features.5': 'cnn_4',
    'features.7': 'cnn_5',
    'features.9': 'cnn_6',
    'features.10': 'cnn_7',
    'features.12': 'cnn_8',
    'features.14': 'cnn_9',
    'features.16': 'cnn_10',
    'features.17': 'cnn_11',
    'features.19': 'cnn_12',
    'features.21': 'cnn_13',
    'features.23': 'cnn_14',
    'features.24': 'cnn_15',
    'features.26': 'cnn_16',
    'features.28': 'cnn_17',
    'features.30': 'cnn_18'
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

for label in tqdm(range(999, 0, -1)):
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



#Xf, Xa = extract_from_X(X)

#L = np.asfortranarray(L, dtype=float)
#Xf = np.asfortranarray(Xf, dtype=float)
#Xa = np.asfortranarray(Xa, dtype=float)

### Compute mu-lasso

#Wf = spams.lasso(L, D=Xf, return_reg_path=False, lambda1=10, mode=0)
#Wa = spams.lasso(L, D=Xa, return_reg_path=False, lambda1=10, mode=0)
