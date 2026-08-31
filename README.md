# Deep Learning 3D Prostate MRI Segmentation 



<p align="center">

&#x20; <img src="Images/MRI Predictions.png" width="850">

</p>

## About this project

This project is my attempt to build a complete deep learning pipeline for **automatic prostate segmentation from 3D MRI scans**.

The goal was simple in theory: give the model an MRI scan and train it to identify the prostate automatically.

In practice, the project involved much more than just training a neural network. I had to work through:

* organizing the dataset correctly
* pairing images with segmentation masks
* converting the original `.mhd/.raw` files to NIfTI
* deciding how to split the data
* choosing the right preprocessing for MRI
* dealing with different image sizes and slice counts
* controlling overfitting
* testing different model configurations
* understanding why a model with a good average Dice score could still make obvious mistakes on individual patients

The final result is a working 3D prostate segmentation pipeline built with **MONAI and PyTorch**.

\---

## What I wanted the model to learn

Each MRI scan has a corresponding segmentation mask.

The segmentation contains two classes:

```text
0 = Background
1 = Prostate
```

The task is therefore a binary segmentation problem.

The model receives:

```text
MRI Volume
    ↓
3D U-Net
    ↓
Predicted Prostate Mask
```

The prediction is then compared with the manually provided ground-truth prostate segmentation.

\---

## Dataset source and citation

The MRI data used in this project come from the **PROMISE12 (Prostate MR Image Segmentation 2012) Challenge**, a MICCAI 2012 challenge created to evaluate automatic and semi-automatic prostate segmentation methods on transverse T2-weighted MRI.

PROMISE12 contains MRI data collected from multiple clinical centers, MRI vendors, and acquisition protocols. This makes the dataset useful for testing whether a segmentation model can generalize across variations in prostate appearance and image acquisition.

The full PROMISE12 challenge dataset contains:

* **50 training cases**
* **30 test cases**
* **20 live-challenge cases**

For this project, I used the 50 referenced training cases to create my development split:

```text
40 patients → Training
10 patients → Validation
```

I then used the referenced 30-case PROMISE12 test set for final testing.

The original files were supplied in MetaImage format (`.mhd` + `.raw`). I converted the MRI volumes and segmentation masks to compressed NIfTI (`.nii.gz`) files before preprocessing and training.

The original PROMISE12 challenge page is available at:

https://promise12.grand-challenge.org/

The complete challenge data, including the released segmentation masks, are available through Zenodo:

https://zenodo.org/records/8026660

### Recommended dataset citation

> Litjens, G., Toth, R., van de Ven, W., et al. (2014). \*\*Evaluation of prostate segmentation algorithms for MRI: The PROMISE12 challenge.\*\* \*Medical Image Analysis, 18\*(2), 359–373. https://doi.org/10.1016/j.media.2013.12.002

> \*\*Dataset:\*\* Litjens, G., van Ginneken, B., Huisman, H., van de Ven, W., Hoeks, C., Barratt, D., \& Madabhushi, A. \*\*PROMISE12: Data from the MICCAI Grand Challenge: Prostate MR Image Segmentation 2012.\*\* Zenodo. https://zenodo.org/records/8026660

> \*\*Data availability note:\*\* The original medical images are not redistributed in this GitHub repository. Anyone who wants to reproduce the project should obtain the PROMISE12 dataset from the official challenge/Zenodo source and follow its licensing and citation requirements.

\---

## Dataset setup

The PROMISE12 labeled training dataset contained **50 patients**.

I divided these into:

```text
40 patients → Training
10 patients → Validation
```

A separate provided test dataset contained:

```text
30 patients → Final Testing
```

The final experimental setup was:

|Dataset|Patients|Purpose|
|-|-:|-|
|Training|40|Used by the model to learn|
|Validation|10|Used to monitor generalization and select the best model|
|Test|30|Used for final evaluation|

I found it useful to think about the three sets as:

```text
Training = homework
Validation = practice exam
Testing = final exam
```

The model learns from the training patients. The validation set helps determine whether the model is learning useful patterns instead of simply memorizing the training data.

The test set is kept separate from training and model selection.

\---

## Data organization

I organized the data into separate image and label folders for each split:

```text
Prostate Segmentation Project/
│
├── Train/
│   ├── Images/
│   │   ├── Case00/
│   │   │   └── Case00.nii.gz
│   │   └── ...
│   └── Labels/
│       ├── Case00/
│       │   └── Case00\_segmentation.nii.gz
│       └── ...
│
├── Validation/
│   ├── Images/
│   └── Labels/
│
└── Test/
    ├── Images/
    └── Labels/
```

This made it easier to ensure that every MRI volume was paired with the correct segmentation mask.

> \*\*Note:\*\* The medical imaging dataset is not included in this repository.

\---

## Converting the dataset to NIfTI

The original dataset was provided in MetaImage format:

```text
.mhd
.raw
```

Each MRI depended on two files, so I converted the images to compressed NIfTI format:

```text
.nii.gz
```

For example:

```text
Case00.mhd
Case00.raw

        ↓

Case00.nii.gz
```

The segmentation masks were converted in the same way.

After conversion, I verified that important image geometry information was preserved by comparing:

* image size
* voxel spacing
* origin
* direction

between the original and converted volumes.

\---

## One of the first challenges: different image sizes

The MRI volumes did not all have the same dimensions or number of slices.

For example, different cases could have dimensions such as:

```text
320 × 320 × 20
```

or:

```text
512 × 512 × 23
```

This was a challenge because a neural network requires consistent input dimensions.

Instead of simply deleting slices, I used MONAI transforms to resample and resize the entire 3D volume to a common spatial size.

This became an important part of the preprocessing pipeline.

\---

## MRI preprocessing

Because this project uses MRI rather than CT, fixed Hounsfield Unit windowing was not appropriate.

Instead, I normalized the non-zero MRI intensities:

```python
NormalizeIntensityd(keys=\["image"], nonzero=True, channel\_wise=True)
```

The main preprocessing pipeline included:

```text
LoadImaged
EnsureChannelFirstD
Spacingd
Orientationd
NormalizeIntensityd
CropForegroundd
Resized
ToTensord
```

One of the configurations that performed well used:

```text
128 × 128 × 15
```

with standardized voxel spacing of:

```text
1.0 × 1.0 × 2.0 mm
```

For the segmentation labels, nearest-neighbor interpolation was used so that the mask remained discrete:

```text
0 = Background
1 = Prostate
```

\---

## Data augmentation

With only 40 training patients, overfitting was an important concern.

I applied random augmentation only to the training dataset.

The transformations included:

* random flips
* 90-degree rotations
* intensity shifting
* intensity scaling

The validation and test datasets were kept deterministic and were not randomly augmented.

\---

## The model

I used a **3D U-Net** implemented with MONAI.

A representative model configuration was:

```python
model = UNet(
    spatial\_dims=3,
    in\_channels=1,
    out\_channels=2,
    channels=(16, 32, 64, 128),
    strides=((2, 2, 1), (2, 2, 1), (2, 2, 1)),
    num\_res\_units=2,
    norm=Norm.BATCH,
    dropout=0.2
).to(device)
```

One important design decision was using anisotropic strides:

```text
(2, 2, 1)
```

instead of:

```text
(2, 2, 2)
```

The Z dimension only contained 15 slices after preprocessing. Standard 3D downsampling changed the output depth and caused a shape mismatch between the model output and the segmentation label.

Using `(2, 2, 1)` allowed the network to downsample the X and Y dimensions while preserving the slice dimension.

For example:

```text
256 × 256 × 15
        ↓
128 × 128 × 15
        ↓
64 × 64 × 15
```

This solved the mismatch problem.

\---

## Training

The model was trained using the 40 training patients and evaluated after each epoch using the 10 validation patients.

The main values I monitored were:

```text
Training Loss
Training Dice
Validation Loss
Validation Dice
```

Loss measures how wrong the model is:

```text
Lower = better
```

Dice measures how much the predicted prostate overlaps the ground-truth prostate:

```text
Higher = better
```

A Dice score ranges from:

```text
0 = no overlap
1 = perfect overlap
```

\---

## Overfitting

One of the clearest challenges during the project was overfitting.

In some experiments, the training Dice continued to increase while the validation Dice stopped improving.

This meant the model was becoming very good at segmenting the patients it had already seen but was not improving on unseen patients.

To reduce overfitting, I used:

* data augmentation
* dropout
* weight decay
* learning-rate scheduling
* early stopping

The best model checkpoint was saved whenever the validation Dice improved.

Training was stopped when there was no validation improvement for a predefined number of epochs.

\---

# Model experiments

A major part of this project was experimenting with multiple configurations rather than assuming that the first model would be the best.

I compared four main model experiments.

\---

## Model 1 — Higher-resolution baseline

Model 1 used a larger spatial size:

```text
256 × 256 × 15
```

It served as the initial baseline.

Approximate results:

```text
Validation Dice ≈ 0.59
Test Dice ≈ 0.74
```

The model could often identify the prostate, but it showed weaker generalization and produced several false-positive and over-segmented regions in difficult cases.

One important lesson from this model was that higher input resolution does not automatically lead to better segmentation performance.

\---

## Model 2 — Selected baseline

Model 2 used:

```text
128 × 128 × 15
```

along with spacing normalization, dropout, and weight decay.

Approximate performance:

```text
Validation Dice ≈ 0.62
Test Dice ≈ 0.80
```

Model 2 became my main baseline because it produced strong overall performance and performed best on the largest number of individual test patients among the initial models.

Its main strengths were:

* good prostate localization
* strong overall Dice overlap
* improved generalization compared with Model 1
* lower computational cost

Its main weaknesses were:

* over-segmentation
* occasional false positives
* imperfect prostate boundaries
* difficulty with small prostate regions near the apex and base

\---

## Model 3 — Additional configuration experiment

Model 3 tested another variation of the same general 3D U-Net pipeline.

Its validation performance was competitive, but its overall test performance did not clearly outperform Model 2.

Approximate results were:

```text
Validation Dice ≈ 0.64
Test Dice ≈ 0.79
```

This experiment reinforced an important lesson: a higher validation Dice does not always guarantee better final test performance.

\---

## Model 4 — Loss-function experiments

Model 4 focused mainly on changing the loss function.

I tested:

* DiceCE Loss
* Tversky Loss

### DiceCE Loss

DiceCE combines Dice Loss and Cross Entropy.

The idea was to improve both overlap and voxel-level classification.

However, the final test performance was lower than Model 2, so I did not keep it as the main configuration.

### Tversky Loss

I also tested Tversky Loss because several predictions showed false positives and over-segmentation.

A balanced configuration used:

```text
alpha = 0.5
beta  = 0.5
```

This produced one of the strongest validation Dice scores:

```text
Validation Dice ≈ 0.70
Test Dice ≈ 0.80
```

I then increased the false-positive penalty:

```text
alpha = 0.7
beta  = 0.3
```

The resulting test Dice remained close to Model 2.

Although the Tversky experiments were useful, they did not produce a large enough improvement for me to replace Model 2 as the main baseline.

\---

## Model comparison

|Model|Main change|Approx. Validation Dice|Approx. Test Dice|Observation|
|-|-|-:|-:|-|
|Model 1|256 × 256 × 15 baseline|0.59|0.74|Weaker generalization|
|**Model 2**|**128 × 128 × 15 + spacing normalization**|**0.62**|**0.80**|**Selected baseline**|
|Model 3|Additional configuration|0.64|0.79|Competitive but not better overall|
|Model 4|DiceCE / Tversky experiments|up to \~0.70|\~0.80|Strong validation but little test improvement|

Model 2 remains my selected baseline because it provided a strong balance between test performance, patient-level consistency, and model simplicity.

\---

## Patient-level evaluation

Instead of looking only at the average test Dice, I calculated a Dice score for every test patient.

For example:

```text
Case00 → Dice
Case01 → Dice
Case02 → Dice
...
Case29 → Dice
```

This was one of the most useful parts of the project.

It showed that some patients were consistently easier to segment while others were difficult across several models.

Examples of difficult cases included:

```text
Case24
Case15
Case08
Case10
Case29
```

Looking at these cases helped me understand why a model failed rather than relying only on a single average score.

\---

## What the models get wrong

Visual inspection showed several recurring segmentation errors.

### 1\. Over-segmentation

This was one of the most common errors.

The model usually found the correct prostate location but sometimes predicted a region that was too large.

```text
Ground Truth:
    █████

Prediction:
  █████████
```

The prostate location may be correct while the boundary is inaccurate.

### 2\. False positives

In some slices, the ground truth contained no prostate but the model still produced a prostate prediction.

```text
Ground Truth = empty
Prediction   = prostate region
```

### 3\. Disconnected false-positive regions

Some predictions contained the main prostate region plus a second unrelated predicted structure.

This suggested that the model sometimes confused nearby anatomy with prostate tissue.

### 4\. Apex and base errors

The prostate becomes smaller near the beginning and end of the gland.

These slices were often more difficult for the model.

The model could:

* miss a very small prostate region
* predict prostate when none was present
* overestimate the size of the prostate

\---

## Largest connected component experiment

Because some predictions contained disconnected false-positive regions, I tested **largest connected component (LCC)** post-processing.

The idea was:

```text
Find all predicted regions
        ↓
Keep only the largest one
        ↓
Remove smaller isolated regions
```

Initially this appeared promising visually.

However, the validation results were:

```text
Before LCC ≈ 0.6205
After LCC  ≈ 0.6044
```

The overall validation Dice decreased.

This showed that some smaller components may contain legitimate prostate predictions, particularly near small anatomical regions.

I therefore did not use LCC as a universal post-processing step.

This experiment taught me an important lesson:

> A post-processing method that improves one patient does not necessarily improve the dataset as a whole.

\---

## What I learned

This project helped me understand that medical image segmentation is much more than training a neural network.

A large part of the work involved:

* understanding the dataset
* organizing medical imaging files correctly
* preprocessing MRI appropriately
* avoiding data leakage
* selecting suitable image dimensions
* understanding 3D U-Net behavior
* controlling overfitting
* comparing multiple models
* analyzing individual patient failures
* combining quantitative metrics with visual inspection

One of the biggest lessons was:

> A high training Dice score does not mean the model is good.

Validation and test performance are more important because they show how well the model generalizes to patients it has never seen.

Another important lesson was:

> A prediction can look visually good on one slice and still have a poor 3D Dice score.

For this reason, I used both numerical evaluation and slice-by-slice visual inspection.

\---

## Current result

My selected baseline, Model 2, achieved approximately:

```text
Mean Test Dice ≈ 0.80
```

on the separate 30-patient test dataset.

The model generally localizes the prostate well but still has room for improvement in boundary precision, false-positive suppression, and segmentation of small prostate regions.

I consider Model 2 a **strong baseline rather than a finished clinical model**.

\---

## Future work

The next stages I would like to investigate include:

* 5-fold cross-validation
* prostate-centered cropping
* Instance Normalization
* improved data augmentation
* boundary-aware loss functions
* Hausdorff Distance (HD95)
* sensitivity and specificity
* nnU-Net comparison
* Attention U-Net
* larger datasets
* external validation
* improved handling of false-positive regions

I am particularly interested in improving segmentation at the prostate boundaries and reducing false positives without sacrificing sensitivity at the prostate apex and base.

\---

## Repository structure

A suggested structure for this repository is:

```text
Deep-Learning-3D-Prostate-MRI-Segmentation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── preprocessing/
│   ├── preprocess.py
│   └── Data\_CleanUp.ipynb
│
├── utilities/
│   └── Utilities.py
│
├── models/
│   ├── model\_1/
│   ├── model\_2/
│   ├── model\_3/
│   └── model\_4/
│
└── results/
    └── figures/
        ├── training\_curves/
        ├── model\_predictions/
        ├── difficult\_cases/
        └── model\_comparison/
```

\---

## Technologies used

* Python
* PyTorch
* MONAI
* SimpleITK
* NumPy
* Pandas
* Matplotlib
* SciPy
* Jupyter Notebook

\---

## Reproducibility

To improve reproducibility, deterministic behavior can be enabled with:

```python
set\_determinism(seed=0)
```

The repository does not contain the original medical imaging dataset or patient data.

\---

## References

1. Litjens, G., Toth, R., van de Ven, W., et al. (2014). **Evaluation of prostate segmentation algorithms for MRI: The PROMISE12 challenge.** *Medical Image Analysis, 18*(2), 359–373. https://doi.org/10.1016/j.media.2013.12.002
2. PROMISE12 Grand Challenge. **MICCAI Grand Challenge: Prostate MR Image Segmentation 2012.** https://promise12.grand-challenge.org/
3. Litjens, G., van Ginneken, B., Huisman, H., van de Ven, W., Hoeks, C., Barratt, D., \& Madabhushi, A. **PROMISE12: Data from the MICCAI Grand Challenge: Prostate MR Image Segmentation 2012.** Zenodo. https://zenodo.org/records/8026660

\---

## Final note

This project is still a work in progress.

Rather than showing only the best performance number, I have kept the different experiments, challenges, and failure analysis because they represent the actual process of developing a medical image segmentation model.

The project helped me understand not only how to train a 3D deep learning model, but also how to evaluate its limitations and make evidence-based decisions about what to improve next.

