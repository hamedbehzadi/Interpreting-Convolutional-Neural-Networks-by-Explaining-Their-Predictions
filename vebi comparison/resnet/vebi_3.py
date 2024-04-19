import torch
import torch.nn
import random
import pickle
from collections import defaultdict
from torchvision import transforms
import torchvision.models as models
import numpy as np
import os
from PIL import Image
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
#import spams
from tqdm import tqdm

random.seed(0)

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

ACCURACY_DICT = dict()

if os.path.exists("accuracy.pickle"):
    with open("accuracy.pickle", "rb") as f:
        ACCURACY_DICT = pickle.load(f)

def size_conversion(filter):
    for i in range(3):
        if filter < 64:
            return f"cnn_{i + 1}", filter
        filter -= 64

    for i in range(3):
        if filter < 128:
            return f"cnn_{i + 1 + 3}", filter
        filter -= 128

    for i in range(4):
        if filter < 256:
            return f"cnn_{i + 1 + 3 + 3}", filter
        filter -= 256

    for i in range(8):
        if filter < 512:
            return f"cnn_{i + 1 + 3 + 3 + 4}", filter
        filter -= 512

    raise ValueError("Invalid filter ID")


def extract_filters(W, all=False):
    label_to_filter = defaultdict(lambda: defaultdict(list))
    for filter, label in zip(*W.nonzero()):
        if all:
            cnn, f = size_conversion(filter)
            label_to_filter[cnn][label].append(f)
        else:
            label_to_filter["cnn_18"][label].append(filter)
    return label_to_filter


def process(filter_block_list, type, name):
    return_layers = {
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

    def set_filter(layer, label):
        def hook(model, input, output):
            output[:, filter_block_list[return_layers[layer]][label]] *= 0
            return output

        return hook

    results = defaultdict(list)
    nn = torch.nn.Softmax(dim=1)
    nn.cuda()
    for label in tqdm(range(1000)):
        # Pretrained model
        model = models.resnet50(pretrained=True)
        model.eval()
        model.cuda()

        if type in ["final", "association rules", "random", "grouping"]:
            model.layer4[2].register_forward_hook(set_filter("layer4.2", label))
        elif type == "all":
            pass # TODO
        elif type == "base":
            pass  # no adjustments
        else:
            raise NotImplementedError

        images = []
        def process_batch(images):
            with torch.no_grad():
                images = torch.stack(images).cuda()
                output = model(images)
                _, preds = torch.max(output, 1)
                for i in preds:
                    if i == label:
                        if name not in ACCURACY_DICT:
                            ACCURACY_DICT[name] = np.zeros(1000)
                        ACCURACY_DICT[name][label] += 1
                results[label].extend(nn(output).cpu().numpy()[:, label])

        for image_name in os.listdir(f"../data/val_set/{label}/img"):
            images.append(normalize(t(Image.open(f"../data/val_set/{label}/img/{image_name}").convert('RGB'))))
            if len(images) >= 25:
                process_batch(images)
                images = []
        if len(images) > 0:
            process_batch(images)
            images = []
    return results


if __name__ == "__main__":
    ### Final layer vebi
    with open("filters/Wf.pickle", "rb") as f:
        filters_f = extract_filters(pickle.load(f))

    if os.path.exists(f"scores/final.pickle"):
        with open("scores/final.pickle", "rb") as f:
            scores_final = pickle.load(f)
    else:
        scores_final = process(filters_f, type="final", name="final")
        with open("scores/final.pickle", "wb") as f:
            pickle.dump(scores_final, f)

    ### Baseline
    if os.path.exists(f"scores/baseline.pickle"):
        with open("scores/baseline.pickle", "rb") as f:
            scores_base = pickle.load(f)
    else:
        scores_base = process(None, type="base", name="base")
        with open("scores/baseline.pickle", "wb") as f:
            pickle.dump(scores_base, f)

    ### Random
    if os.path.exists(f"filters/random.pickle"):
        with open("filters/random.pickle", "rb") as f:
            filters_random = pickle.load(f)
    else:
        filters_random = {"cnn_18": [random.sample(range(512), 10) for _ in range(1000)]}
        with open("filters/random.pickle", "wb") as f:
            pickle.dump(filters_random, f)


    if os.path.exists(f"scores/random.pickle"):
        with open("scores/random.pickle", "rb") as f:
            scores_random = pickle.load(f)
    else:
        scores_random = process(filters_random, type="random", name="random")
        with open("scores/random.pickle", "wb") as f:
            pickle.dump(scores_random, f)


    ### Assoc Rule Based
    with open("filters/assoc.pickle", "rb") as f:
        filters_assoc = pickle.load(f)

    if os.path.exists(f"scores/assoc.pickle"):
        with open("scores/assoc.pickle", "rb") as f:
            scores_assoc = pickle.load(f)
    else:
        scores_assoc = process(filters_assoc, type="association rules", name="assoc")
        with open("scores/assoc.pickle", "wb") as f:
            pickle.dump(scores_assoc, f)


    ### Assoc Rule Based (limit = 5)
    with open("filters/assoc5.pickle", "rb") as f:
        filters_assoc_5 = pickle.load(f)

    if os.path.exists(f"scores/assoc5.pickle"):
        with open("scores/assoc5.pickle", "rb") as f:
            scores_assoc_5 = pickle.load(f)
    else:
        scores_assoc_5 = process(filters_assoc_5, type="association rules", name="assoc5")
        with open("scores/assoc5.pickle", "wb") as f:
            pickle.dump(scores_assoc_5, f)

    ### Assoc Rule Based (more than 5 long transactoins)
    with open("filters/assoc_larger.pickle", "rb") as f:
        filters_assoc_L= pickle.load(f)

    if os.path.exists(f"scores/assoc_larger.pickle"):
        with open("scores/assoc_larger.pickle", "rb") as f:
            scores_assoc_L = pickle.load(f)
    else:
        scores_assoc_L = process(filters_assoc_L, type="association rules", name="assocL")
        with open("scores/assoc_larger.pickle", "wb") as f:
            pickle.dump(scores_assoc_L, f)

    ### Full System Based
    with open("filters/main.pickle", "rb") as f:
        filters_main = {"cnn_18": pickle.load(f)}

    if os.path.exists(f"scores/main.pickle"):
        with open("scores/main.pickle", "rb") as f:
            scores_main = pickle.load(f)
    else:
        scores_main = process(filters_main, type="grouping", name="main")
        with open("scores/main.pickle", "wb") as f:
            pickle.dump(scores_main, f)


    ### accuracy
    with open("accuracy.pickle", "wb") as f:
        pickle.dump(ACCURACY_DICT, f)

    print("Loaded all scores and filters...")
    len_dict = defaultdict(np.ndarray)
    for i in range(1000):
        len_dict["final"] = np.asarray([len(filters_f["cnn_18"][i]) for i in range(1000)])
        len_dict["assoc"] = np.asarray([len(filters_assoc["cnn_18"][i]) for i in range(1000)])
        len_dict["assoc5"] = np.asarray([len(filters_assoc_5["cnn_18"][i]) for i in range(1000)])
        len_dict["assocL"] = np.asarray([len(filters_assoc_L["cnn_18"][i]) for i in range(1000)])
        len_dict["random"] = np.asarray([len(filters_random["cnn_18"][i]) for i in range(1000)])
        len_dict["main"] = np.asarray([len(filters_main["cnn_18"][i]) for i in range(1000)])
        len_dict["base"] = np.zeros(1000)

    """
        scores = defaultdict(lambda: defaultdict(np.ndarray))

        for i in range(1000):
            scores["final"][i] = np.abs(np.asarray(scores_base[i]) - np.asarray(scores_final[i]))
            scores["assoc"][i] = np.abs(np.asarray(scores_base[i]) - np.asarray(scores_assoc[i]))
            scores["assoc5"][i] = np.abs(np.asarray(scores_base[i]) - np.asarray(scores_assoc_5[i]))
            scores["assocL"][i] = np.abs(np.asarray(scores_base[i]) - np.asarray(scores_assoc_L[i]))
            scores["random"][i] = np.abs(np.asarray(scores_base[i]) - np.asarray(scores_random[i]))
            scores["main"][i] = np.abs(np.asarray(scores_base[i]) - np.asarray(scores_main[i]))
            scores["base"] = np.zeros(1000)

        scores2 = defaultdict(lambda: defaultdict(lambda: np.zeros(1000)))
        for key in scores:
            for i in range(1000):
                scores2[key]["std"][i] = np.std(scores[key][i])
                scores2[key]["median"][i] = np.median(scores[key][i])
                scores2[key]["mean"][i] = np.mean(scores[key][i])
                scores2[key]["min"][i] = np.min(scores[key][i])
                scores2[key]["max"][i] = np.max(scores[key][i])
                scores2[key]["acc"][i] = ACCURACY_DICT[key][i] / 50
            scores2[key]["len"] = len_dict[key]

        scores3 = defaultdict(lambda: defaultdict(float))
        for key in scores2:
            for metric in scores2[key]:
                scores3[key][metric] = np.mean(scores2[key][metric])
    """
    scores3 = defaultdict(lambda: defaultdict(float))
    for key in len_dict:
        print(key)
        scores3[key]["mean_acc"] = np.mean(ACCURACY_DICT[key]/ 50)
        scores3[key]["sd_acc"] = np.std(ACCURACY_DICT[key] / 50)
        scores3[key]["medi_acc"] = np.median(ACCURACY_DICT[key] /50)
        scores3[key]["mean_len"] = np.mean(len_dict[key])
        scores3[key]["std_len"] = np.std(len_dict[key])
        scores3[key]["medi_len"] = np.median(len_dict[key])


    print("=" * 30)
    for key in scores3:
        for metric in scores3[key]:
            if metric in ["medi_len", "mean_len"]:
                print(f"{key},\t{metric}:\t{scores3[key][metric]:.2f}")
            else:
                print(f"{key},\t{metric}:\t{scores3[key][metric]}")
        print("=" * 30)

    dif = ACCURACY_DICT["assoc"] - ACCURACY_DICT["final"]
    print(np.flatnonzero(dif == np.max(dif)))
    print(np.flatnonzero(dif == np.min(dif)))
    print(np.flatnonzero(ACCURACY_DICT["assoc"] == np.min(ACCURACY_DICT["assoc"])))
    dif = ACCURACY_DICT["assoc"] - ACCURACY_DICT["base"]
    print(np.flatnonzero(dif == np.max(dif)))
    print(np.flatnonzero(ACCURACY_DICT["assoc"] > ACCURACY_DICT["base"]))
    print(np.flatnonzero(ACCURACY_DICT["final"] > ACCURACY_DICT["base"]))
    print(set.intersection(set(np.flatnonzero(ACCURACY_DICT["assoc"] > ACCURACY_DICT["base"])), set(np.flatnonzero(ACCURACY_DICT["final"] > ACCURACY_DICT["base"]))))
    print(np.flatnonzero(ACCURACY_DICT["main"] > ACCURACY_DICT["base"]))
    print(set.intersection(set(np.flatnonzero(ACCURACY_DICT["main"] > ACCURACY_DICT["base"])), set(np.flatnonzero(ACCURACY_DICT["final"] > ACCURACY_DICT["base"]))))
    print(set.intersection(set(np.flatnonzero(ACCURACY_DICT["assoc"] > ACCURACY_DICT["base"])), set(np.flatnonzero(ACCURACY_DICT["main"] > ACCURACY_DICT["base"]))))


    '''
    for key in len_dict:
        print(f"\t{key}", end="")
    print()
    for i in sorted(range(1000)):
        print(f"{i}:\t", end="")
        for key in len_dict:
            print(f"{ACCURACY_DICT[key][i] * 2}%\t", end="")
        print()
    '''