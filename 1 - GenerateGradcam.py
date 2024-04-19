from utils.mapping import image_mapping
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import torchvision.models as models
import pickle
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
import shutil

if __name__ == "__main__":
    DEBUG = False

    # Pretrained model
    model = models.vgg16(pretrained=True)
    model.eval()
    model.cuda()
    target_layers = [model.features[-1]]

    # values to normalize input
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    normalize = transforms.Normalize(mean, std)

    with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:
        for filename in tqdm(os.listdir("data/Images")):
            # Check if this file has already been processed
            if not DEBUG:
                with open('finished.txt', 'r') as f:
                    cnt = False
                    for line in f.readlines():
                        if filename == line[:-1]:
                            cnt = True
                            break
                    if cnt:
                        continue

            # Get label
            true_label_id = image_mapping[filename]

            # Ensure label dir exists
            if not os.path.exists('data/val_set/' + str(true_label_id) + "/"):
                os.makedirs('data/val_set/' + str(true_label_id) + "/img/")
                os.makedirs('data/val_set/' + str(true_label_id) + "/grad/")

            # Read file
            input = t(Image.open("data/Images/" + filename).convert('RGB'))[None, :, :].cuda()

            # Copy file to working dir
            if not DEBUG:
                shutil.copy2('data/Images/' + filename, 'data/val_set/' + str(true_label_id) + "/img/")

            # Generate cam
            grayscale_cam = cam(input_tensor=normalize(input))
            grayscale_cam = grayscale_cam[0, :]

            # Store cam
            if not DEBUG:
                pickle.dump(grayscale_cam,
                            open('data/val_set/' + str(true_label_id) + '/grad/' + filename[:-5] + ".grad", "wb"))

            # Mark file as finished to not rerun everything in case of crash
            if not DEBUG:
                with open('finished.txt', 'a') as f:
                    f.write(filename + '\n')

            # Extra info for debug
            if DEBUG:
                visualization = show_cam_on_image(np.transpose(input.data[0].cpu().numpy(), (1, 2, 0))
                                                  , grayscale_cam, use_rgb=True)

                test = input.data[0].cpu().numpy()
                test = np.transpose(test, (1, 2, 0))
                fig, axs = plt.subplots(2)
                axs[0].imshow(test)
                axs[1].imshow(visualization)
                plt.show()
                break
