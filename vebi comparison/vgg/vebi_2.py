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
Xf = []
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
        Xf.append(item["cnn_18"])

L = np.asfortranarray(L, dtype=float)
Xf = np.asfortranarray(Xf, dtype=float)
Xa = np.asfortranarray(Xa, dtype=float)

### Compute mu-lasso
print("Fortran")

Wf = spams.lasso(L, D=Xf, return_reg_path=False, verbose=True, pos=True, lambda1=10, mode=0)
with open("Wf.pickle", "wb") as f:
    pickle.dump(Wf, f)
del Wf
del Xf

Wa = spams.lasso(L, D=Xa, return_reg_path=False, verbose=True, pos=True, lambda1=10, mode=0)
with open("Wa.pickle", "wb") as f:
    pickle.dump(Wa, f)