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
        model = models.vgg16(pretrained=True)
        model.eval()
        model.cuda()

        if type in ["final", "association rules", "random", "grouping"]:
            model.features[30].register_forward_hook(set_filter("features.30", label))
        elif type == "all":
            model.features[0].register_forward_hook(set_filter("features.0", label))
            model.features[2].register_forward_hook(set_filter("features.2", label))
            model.features[4].register_forward_hook(set_filter("features.4", label))
            model.features[5].register_forward_hook(set_filter("features.5", label))
            model.features[7].register_forward_hook(set_filter("features.7", label))
            model.features[9].register_forward_hook(set_filter("features.9", label))
            model.features[10].register_forward_hook(set_filter("features.10", label))
            model.features[12].register_forward_hook(set_filter("features.12", label))
            model.features[14].register_forward_hook(set_filter("features.14", label))
            model.features[16].register_forward_hook(set_filter("features.16", label))
            model.features[17].register_forward_hook(set_filter("features.17", label))
            model.features[19].register_forward_hook(set_filter("features.19", label))
            model.features[21].register_forward_hook(set_filter("features.21", label))
            model.features[23].register_forward_hook(set_filter("features.23", label))
            model.features[24].register_forward_hook(set_filter("features.24", label))
            model.features[26].register_forward_hook(set_filter("features.26", label))
            model.features[28].register_forward_hook(set_filter("features.28", label))
            model.features[30].register_forward_hook(set_filter("features.30", label))
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

    ### accuracy
    with open("accuracy.pickle", "wb") as f:
        pickle.dump(ACCURACY_DICT, f)

    ### All layer vebi
    with open("filters/Wa.pickle", "rb") as f:
        filters_a = extract_filters(pickle.load(f), all=True)

    if os.path.exists(f"scores/all.pickle"):
        with open("scores/all.pickle", "rb") as f:
            scores_all = pickle.load(f)
    else:
        scores_all = process(filters_a, type="all", name="all")
        with open("scores/all.pickle", "wb") as f:
            pickle.dump(scores_all, f)

    ### accuracy
    with open("accuracy.pickle", "wb") as f:
        pickle.dump(ACCURACY_DICT, f)

    ### Baseline
    if os.path.exists(f"scores/baseline.pickle"):
        with open("scores/baseline.pickle", "rb") as f:
            scores_base = pickle.load(f)
    else:
        scores_base = process(None, type="base", name="base")
        with open("scores/baseline.pickle", "wb") as f:
            pickle.dump(scores_base, f)

    ### accuracy
    with open("accuracy.pickle", "wb") as f:
        pickle.dump(ACCURACY_DICT, f)

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

    ### accuracy
    with open("accuracy.pickle", "wb") as f:
        pickle.dump(ACCURACY_DICT, f)

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

    ### accuracy
    with open("accuracy.pickle", "wb") as f:
        pickle.dump(ACCURACY_DICT, f)

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
            
    ### accuracy
    with open("accuracy.pickle", "wb") as f:
        pickle.dump(ACCURACY_DICT, f)

    ### Assoc Rule Based (large)
    with open("filters/assocL.pickle", "rb") as f:
        filters_assoc_l = pickle.load(f)

    if os.path.exists(f"scores/assocL.pickle"):
        with open("scores/assocL.pickle", "rb") as f:
            scores_assoc_l = pickle.load(f)
    else:
        scores_assoc_l = process(filters_assoc_l, type="association rules", name="assocL")
        with open("scores/assocL.pickle", "wb") as f:
            pickle.dump(scores_assoc_l, f)

    ### accuracy
    with open("accuracy.pickle", "wb") as f:
        pickle.dump(ACCURACY_DICT, f)

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
        len_dict["all"] = np.sum(np.asarray([[len(filters_a[key][i]) for i in range(1000)] for key in filters_a]), 0)
        len_dict["final"] = np.asarray([len(filters_f["cnn_18"][i]) for i in range(1000)])
        len_dict["assoc"] = np.asarray([len(filters_assoc["cnn_18"][i]) for i in range(1000)])
        len_dict["assoc5"] = np.asarray([len(filters_assoc_5["cnn_18"][i]) for i in range(1000)])
        len_dict["assocL"] = np.asarray([len(filters_assoc_l["cnn_18"][i]) for i in range(1000)])
        len_dict["random"] = np.asarray([len(filters_random["cnn_18"][i]) for i in range(1000)])
        len_dict["main"] = np.asarray([len(filters_main["cnn_18"][i]) for i in range(1000)])
        len_dict["base"] = np.zeros(1000)

    """
    scores = defaultdict(lambda: defaultdict(np.ndarray))

    for i in range(1000):
        scores["all"][i] = np.abs(np.asarray(scores_base[i]) - np.asarray(scores_all[i])) / np.asarray(scores_base[i])
        scores["final"][i] = np.abs(np.asarray(scores_base[i]) - np.asarray(scores_final[i])) / np.asarray(scores_base[i])
        scores["assoc"][i] = np.abs(np.asarray(scores_base[i]) - np.asarray(scores_assoc[i])) / np.asarray(scores_base[i])
        scores["assoc5"][i] = np.abs(np.asarray(scores_base[i]) - np.asarray(scores_assoc_5[i])) / np.asarray(scores_base[i])
        scores["random"][i] = np.abs(np.asarray(scores_base[i]) - np.asarray(scores_random[i])) / np.asarray(scores_base[i])
        scores["main"][i] = np.abs(np.asarray(scores_base[i]) - np.asarray(scores_main[i])) / np.asarray(scores_base[i])
        scores["base"] = np.zeros(1000)

    scores2 = defaultdict(lambda: defaultdict(lambda: np.zeros(1000)))
    for key in scores:
        for i in range(1000):
            scores2[key]["std"][i] = np.std(scores[key][i])
            scores2[key]["min"][i] = np.min(scores[key][i])
            scores2[key]["median"][i] = np.median(scores[key][i])
            scores2[key]["mean"][i] = np.mean(scores[key][i])
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
        scores3[key]["mean_acc"] = np.mean(ACCURACY_DICT[key] / 50)
        scores3[key]["std_acc"] = np.std(ACCURACY_DICT[key] / 50)
        scores3[key]["medi_acc"] = np.median(ACCURACY_DICT[key] / 50)
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
