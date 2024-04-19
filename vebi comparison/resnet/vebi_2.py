from torchvision import transforms
import torchvision.models as models
import numpy as np
import os
import pickle
from PIL import Image
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
import spams
from tqdm import tqdm


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

L = []
Xa = []
#Xf = []
for filename in tqdm(os.listdir("pickle")):
    with open(f"pickle/{filename}", "rb") as f:
        X = pickle.load(f)

    label = int(filename[:-9])
    L.extend([binary(label)] * len(X))
    for item in X:
        tmp = []
        for key in item:
            tmp.extend(item[key])
        Xa.append(tmp)
        #Xf.append(item["cnn_18"])

L = np.asfortranarray(L, dtype=float)
#Xf = np.asfortranarray(Xf, dtype=float)
Xa = np.asfortranarray(Xa, dtype=float)

### Compute mu-lasso
print("Fortran")

#Wf = spams.lasso(L, D=Xf, return_reg_path=False, verbose=True, pos=True, lambda1=10, mode=0)
#with open("filters/Wf.pickle", "wb") as f:
#    pickle.dump(Wf, f)
#del Wf
#del Xf

Wa = spams.lasso(L, D=Xa, return_reg_path=False, verbose=True, pos=True, lambda1=10, mode=0)
with open("filters/Wa.pickle", "wb") as f:
    pickle.dump(Wa, f)
