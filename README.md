# Interpreting-Convolutional-Neural-Networks-by-Explaining-Their-Predictions
# Abstract
We propose a method that exploits the feedback provided by visual explanation methods combined with pattern mining techniques to identify the relevant class-specific and class-shared internal units. In addition, we put forward a patch extraction approach to find faithfully class-specific and class-shared visual patterns. Contrary to the common practice in literature, our approach does not require pushing augmented visual patches through the model. Experiments on two CNN architectures show the effectiveness of the proposed method.


## Description of the proposed interpretation pipline
Proposed Interpretation method. (1) Visual explanations are generated by an explanation method, followed by (2) extraction of patches showing highlighted parts. (3) A transaction dataset is created via the patches. Finally, a pattern mining algorithm (4-6) extracts class-specific and class-shared convolutional filters and their corresponding visual patches. This serves as a means for model interpretation.

![Revised_teaser_figure drawio](https://github.com/hamedbehzadi/Interpreting-Convolutional-Neural-Networks-by-Explaining-Their-Predictions/assets/45251957/4c22e848-18ed-4df6-a85e-bb3ad476425e)

# Evaluation
## Quantitative evaluation
We consider variants of our method where 5, 10, and 25 top filter indices are selected as the elements of the transactions. As can be seen, Ours-10 leads to the lowest model accuracy in VGG16 and ResNet50 showing the efficiency of the proposed method in identifying higher number of filters.

![Screenshot from 2024-04-19 10-43-52](https://github.com/hamedbehzadi/Interpreting-Convolutional-Neural-Networks-by-Explaining-Their-Predictions/assets/45251957/e2caec42-3437-4d8d-801c-04e17f42819e)

## Qualitative Evaluation
### Examplar image per layer and filter
![Screenshot from 2024-04-19 10-46-37](https://github.com/hamedbehzadi/Interpreting-Convolutional-Neural-Networks-by-Explaining-Their-Predictions/assets/45251957/49bd7ee8-f95c-4836-aa63-ae0418a8021e)

### Average of receiptive feild visualiztions per layer and filter for each class
![Screenshot from 2024-04-19 10-46-11](https://github.com/hamedbehzadi/Interpreting-Convolutional-Neural-Networks-by-Explaining-Their-Predictions/assets/45251957/2904839f-0e65-43c0-9083-df6a91b1f381)

# Implemented Code

