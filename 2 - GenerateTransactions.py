from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import torchvision.models as models
import torch
import pickle
import numpy as np
import scipy.ndimage
from scipy import interpolate
from tqdm import tqdm
import os
from PIL import Image
import time
import matplotlib.pyplot as plt
import numpy.random as R
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter


def extract_patches(mask: np.ndarray, image_name: str, input, lv_pair):
    return extract_patches_mc(mask, image_name, input, lv_pair)


def extract_patches_basic(mask: np.ndarray, image_name: str, input, lv_pair):
    """

    :param mask: The cam mask.
    :returns: ndarray of patches
    """
    window_amount = 4  # effect of these parameters
    window_size = int(224 / window_amount)
    stride = int(window_size / 4)
    actual_window_amount = int(224 / stride)

    patches = []
    # extend for multiple objects --> clustering algorithms?
    # look for example of multi objects
    # go from patch to heatmap to internal unit
    # which conv filter is responsible for a specific patch
    for x in range(actual_window_amount):
        for y in range(actual_window_amount):
            if x * stride + window_size > 224 or y * stride + window_size > 224:
                continue
            ws = x * stride
            we = x * stride + window_size
            hs = y * stride
            he = y * stride + window_size

            patches.append((np.sum(mask[hs:he, ws:we]), (hs, he, ws, we),
                            [],  # cnn indices
                            {"image_name": image_name}))  # image name

    return sorted(patches, reverse=True, key=lambda x: x[0])[:25]
    # use weight vector of heatmap to cluster them to identify filter to represent projection of filter on image
    # patches as final step then


def extract_patches_mc(mask: np.ndarray, image_name: str, input, lv_pair):
    """

    :param mask: The cam mask.
    :returns: ndarray of patches
    """

    iterations = 1000

    patches = []
    # Monte Carlo version
    for i in range(10):
        tmp = []
        for _ in range(iterations):
            top, left = R.randint(1, 224 - 56, 2)
            size = 56
            hs = top
            ws = left
            he = top + size
            we = left + size

            tmp.append((np.mean(mask[hs:he, ws:we]), (hs, he, ws, we), image_name))  # image name
        top = sorted(tmp, reverse=True, key=lambda x: x[0])[0]

        # mask[top[1][0] + 10:top[1][1] - 10, top[1][2] + 10:top[1][3] - 10] /= 2
        mask[top[1][0] + 10:top[1][1] - 10, top[1][2] + 10:top[1][3] - 10] = 0
        patches.append({"mask_score": top[0], "mask": top[1], "name": top[2]})

    images = []
    for p in patches:
        image = input.clone()
        image[:, :, p["mask"][0]:p["mask"][1], p["mask"][2]:p["mask"][3]] = 0
        images.append(image[0])

    image2 = input.clone()
    for p in patches:
        image2[:, :, p["mask"][0]:p["mask"][1], p["mask"][2]:p["mask"][3]] = 0
        images.append(image2[0].clone())

    images = torch.stack(images)
    with torch.no_grad():
        images = images.cuda()
        out = model(images)
    for i, p in enumerate(patches):
        p["perturbed"] = out[i]

    to_remove = []
    last = None
    for i, p in enumerate(patches):
        diff = 1 - (out[i + 10][lv_pair[0]] / lv_pair[1])
        if last is not None:
            if diff - last < 0.1:
                return patches[:i]
        last = diff

    return patches


if __name__ == "__main__":
    # accuracy_dict = dict()

    DEBUG = False

    # Pretrained model
    model = models.vgg16(pretrained=True)
    model.eval()

    return_layers = {
        'features.30': 'cnn'
    }
    mid_getter = MidGetter(model, return_layers=return_layers, keep_output=True)

    model = None
    model = models.vgg16(pretrained=True)
    model.eval()
    model.cuda()

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

    for class_id in tqdm(range(1000)):
        # transactions = []

        dir = "data/val_set/" + str(class_id) + "/"
        all_patches = []
        for filename in os.listdir(dir + "img/"):
            # Read file
            input = t(Image.open(dir + "img/" + filename).convert('RGB'))[None, :, :]

            grayscale_cam = pickle.load(open(dir + "grad/" + filename[:-5] + ".grad", "rb"))

            if DEBUG:
                visualization = show_cam_on_image(np.transpose(input.data[0].numpy(), (1, 2, 0))
                                                  , grayscale_cam, use_rgb=True)
                test = input.data[0].numpy()
                test = np.transpose(test, (1, 2, 0))
                fig, axs = plt.subplots(2)
                axs[0].imshow(test)
                axs[1].imshow(visualization)
                plt.savefig(f'plots/{filename}.png')
                plt.close(fig)

            input = normalize(input)
            mid_outputs, model_output = mid_getter(input)
            original_cnn = mid_outputs['cnn'].detach().numpy()
            pred_value, preds = torch.max(model_output, 1)
            # resized_cnn = np.array([scipy.ndimage.interpolation.zoom(input=i, zoom=224. / len(i),
            #                                               order=1) for i in original_cnn[0]])

            resized_cnn = np.array(
                [np.asarray(Image.fromarray(i).resize((224, 224), resample=Image.BILINEAR)) for i in original_cnn[0]])
            # 512, 224, 224

            patches_tmp = extract_patches(grayscale_cam, filename, input, (preds.item(), pred_value.item()))
            for _p_index, p in enumerate(patches_tmp):
                with torch.no_grad():
                    ## return sorted list
                    # cnn_values = np.sum(np.sum(mid_outputs['cnn'].numpy(), axis=2), axis=2)[0]

                    ### for extraction based on location
                    cnn_values = np.sum(np.sum(resized_cnn[:, p["mask"][0]:p["mask"][1], p["mask"][2]:p["mask"][3]],
                                               axis=1), axis=1)

                    ## store results
                    p["features"] = cnn_values  # keep labels for the patches
                    # p["patch_label"] = patch_preds.item()
                    p["perturbed_score"] = 1 - (p["perturbed"] / model_output[0].cuda())
                    del p["perturbed"]
                    p["true_label"] = class_id
                    p["image_label"] = preds.item()
                    # p["image_label_value"] = pred_value.item()

                if DEBUG:
                    plt.imshow(visualization[p["mask"][0]:p["mask"][1], p["mask"][2]:p["mask"][3], :])
                    plt.savefig(f'plots/{filename}_{_p_index}.png')
                    plt.close()

            all_patches.extend(patches_tmp)
            if DEBUG:
                pass
                #break

        if not DEBUG:
            # pickle.dump(transactions, open('data/val_set/' + str(class_id) + '/' + "transactions.pkl", "wb"))
            pickle.dump(all_patches, open("data/val_set/" + str(class_id) + "/patches_interpolated_filters.pkl", "wb"))
            pass

