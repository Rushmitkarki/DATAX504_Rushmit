# Crop Leaf Disease Classification with Transfer Learning

Folder: `final-project/` in my course repository.

DATAX504 final project, University of Waikato. Author: Rushmit Karki (individual).

Classifies crop leaf photos into 38 healthy/diseased classes using a CNN baseline and MobileNetV2 transfer learning (frozen base, then fine-tuned), evaluated on a **leaf-grouped** split so that photos of the same physical leaf never appear in both training and test data.

## Results

Test accuracy on 8,220 held-out images (the test set was used once, after model choices were made on the validation set):

| Model | Validation acc. | Test acc. |
|---|---|---|
| Baseline CNN (from scratch) | 56.6% | 53.9% |
| MobileNetV2, frozen base + new head | 87.8% | 86.2% |
| MobileNetV2, fine-tuned (top layers unfrozen) | 94.9% | **94.5%** |

Exact values are in `results/results.json`.

### Why the split matters

The dataset contains several photos of the same leaf. A plain random split puts near-duplicates in both train and test and inflates scores. An earlier run of this project with a stratified random split gave test accuracies of 88.3% (baseline), 92.2% (frozen) and 96.7% (fine-tuned), so leakage inflated the baseline by about 34 points and the fine-tuned model by about 2 points. Those random-split numbers come from an earlier version of the notebook and are not reproduced by the notebook in this repo.

## Data

- **Source:** the original PlantVillage release, https://github.com/spMohanty/PlantVillage-Dataset (colour images in `raw/color`). The notebook downloads it automatically.
- **Size:** 54,305 image files found, 38 classes (14 crop species). The dataset README states 54,306 images.
- **Citation:** Mohanty, S. P., Hughes, D. P., & Salathe, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419. https://doi.org/10.3389/fpls.2016.01419
- **Licence:** the repository has no licence file. Its README describes the data as an open-access repository and asks users to cite the paper above.

### Split

- Stratified, leaf-grouped 70/15/15 split using scikit-learn `StratifiedGroupKFold` (20 folds: 14 train, 3 validation, 3 test), seed 42.
- Leaf groups come from the repository's `leaf-map.json`. It covers 41,111 images (75.7%) in 7,286 leaves. Images without an entry are treated as single-image groups, so some leakage may remain in those classes (for example the corn classes, grape healthy, squash powdery mildew, tomato target spot and tomato mosaic virus). 4,370 file names map to more than one leaf id; the first id was used.
- Resulting sizes: 38,094 train / 7,991 validation / 8,220 test. The full assignment is saved in `results/split.csv`.

## Method

Images are resized to 160x160; batch size 64; loss is sparse categorical cross-entropy; Adam optimiser.

1. **Baseline CNN:** three Conv2D + MaxPooling blocks (32, 64, 128 filters), global average pooling, dropout 0.3, softmax. 98,150 parameters. Up to 12 epochs, early stopping (patience 3), no data augmentation.
2. **MobileNetV2, frozen base:** ImageNet weights, base frozen, random flip/rotation/zoom augmentation, global average pooling, dropout 0.3, new softmax head. Learning rate 1e-3, up to 6 epochs, early stopping (patience 2).
3. **MobileNetV2, fine-tuned:** the top 54 of the base's 154 layers are unfrozen but BatchNorm layers are kept frozen, so 36 layers train, learning rate 1e-4, up to 8 epochs, early stopping (patience 3).

The best checkpoint by validation loss is kept for each stage. Progress is also backed up every epoch so an interrupted Colab session resumes where it stopped.

## How to reproduce

Developed on Google Colab (T4 GPU, Python 3.13).

1. Open `DATAX504_final_project.ipynb` in Google Colab.
2. Choose **Runtime > Change runtime type > T4 GPU**.
3. Run the cells **in order, one at a time**:
   1. Setup: mounts Google Drive, sets the seed and creates the output folder `DATAX504_final_grouped` on Drive.
   2. Data: downloads PlantVillage, builds the leaf-grouped split and the input pipelines.
   3. Baseline CNN.
   4. MobileNetV2, frozen base.
   5. MobileNetV2, fine-tuning.
   6. Test evaluation and confusion matrix.
   7. Grad-CAM.
   8. Results summary and training curves.
4. Total run time is roughly 40 to 50 minutes. If Colab disconnects, run cells 1 and 2 again and then the cell that was running; training resumes from the last saved epoch.

Dependencies are listed in `requirements.txt`. Results can vary slightly between runs because GPU training is not fully deterministic.

## Repository contents

- `DATAX504_final_project.ipynb`: full pipeline (data, training, evaluation, Grad-CAM)
- `results/`: `results.json`, `split.csv`, per-stage training logs (`*_log.csv`)
- `figures/`: `confusion_matrix.png`, `gradcam.png`, `training_curves.png`
- `requirements.txt`

## Limitations

- PlantVillage photos were taken in controlled conditions on plain backgrounds, so these scores likely overstate field performance.
- The leaf grouping covers about three quarters of the images, so the split is not guaranteed to be leakage-free.
- Single training run per model; per-class results for small classes are noisy.
- The baseline was trained for a fixed 12 epochs without augmentation, so the baseline-to-MobileNetV2 gap is partly an effect of that setup.

## AI assistance

AI tool use is disclosed in the project report.
