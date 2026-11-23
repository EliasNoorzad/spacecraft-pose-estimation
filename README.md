# Spacecraft Pose Estimation

This project investigates autonomous spacecraft navigation by estimating the Hubble Space Telescope’s position and orientation relative to a monocular camera. YOLOv8s-Pose detects 37 landmarks in each image. A Perspective-n-Point (PnP) solver uses their known 3D locations to recover the camera-relative six-degree-of-freedom (6-DoF) pose.

**Latest result.** On the 1,000 held-out images from the 10K experiment, the pipeline returned a pose for every image. Median errors were **0.2326 m** in translation and **2.3508°** in rotation. The largest errors reached **253.8642 m** and **179.5848°**, respectively.

## Dataset and experiments

Both experiments use synthetic Hubble Space Telescope (HST) images from [DLVS3-HST on Zenodo](https://zenodo.org/records/15731476). Each frame has a ground-truth pose and 2D projections of 37 predefined 3D keypoints on the HST model.

| Images | Initial 1K smoke test | Latest 10K experiment |
| --- | ---: | ---: |
| Total | 1,000 | 10,000 |
| Training | 800 | 8,000 |
| Validation | 100 | 1,000 |
| Testing | 100 | 1,000 |

## Methodology

1. **Prepare YOLO pose labels.** The preprocessing script matches each PNG to its annotation. For the 10K run, it reads each numbered image folder with its corresponding metadata JSON, combines the pairs, then makes one global 80/10/10 split using seed 42. It forms a bounding box around valid keypoints, adds 10% padding on each side (at least two pixels per axis), and clips the box to the image. Labels store normalized coordinates in the original landmark order with `kpt_shape: [37, 3]`. A round-trip check verifies the pixel coordinates. Visibility is `v=0` for invalid or out-of-frame points and `v=1` for in-frame points. The PNG annotations cannot reliably distinguish occluded landmarks.
2. **Train the detector.** YOLOv8s-Pose learns to predict all 37 keypoints. The 10K training configuration uses a maximum of 100 epochs, image size 640, batch size 8, seed 42, and early-stopping patience of 25 epochs.
3. **Recover pose.** OpenCV’s iterative PnP solver (`SOLVEPNP_ITERATIVE`) takes all 37 predicted 2D keypoints in pixel coordinates, the corresponding fixed 3D points, and the camera intrinsic matrix. It returns a rotation matrix and a translation vector without RANSAC or per-keypoint confidence filtering.
4. **Evaluate pose.** The evaluation converts the annotated transforms to the OpenCV camera frame and checks their reprojection against the annotated pixels. Translation error is the Euclidean distance between estimated and ground-truth translations. Rotation error is the angle between the estimated and ground-truth orientations.

## Latest results — 10K experiment

The 10K and 1K experiments use different held-out test sets. Comparisons below describe performance on each run’s own test set. They do not isolate the effect of training-set size on identical test images.

### Detection and 2D localization

Box and pose metrics use 1,000 test images for the 10K experiment and 100 for the smoke test. Pose mAP uses a custom object keypoint similarity (OKS) sigma of **0.10** for each landmark.

| Metric | 10K box | 10K pose | 1K box | 1K pose |
| --- | ---: | ---: | ---: | ---: |
| Precision | 0.999 | 0.991 | 1.000 | 0.876 |
| Recall | 1.000 | 0.991 | 0.999 | 0.860 |
| mAP50 | 0.995 | 0.994 | 0.995 | 0.789 |
| mAP50–95 | 0.985 | 0.984 | 0.882 | 0.542 |

Localization error measures the Euclidean distance, in original-image pixels, from each predicted landmark to its valid in-frame annotation on images with a detection.

| Localization metric | Latest 10K | Initial 1K |
| --- | ---: | ---: |
| Images evaluated | 1,000 | 99 |
| Missing detections | 0 | 1 |
| Keypoints evaluated | 36,985 | 3,662 |
| Mean error | 6.382 px | 34.887 px |
| Median error | 4.125 px | 19.527 px |
| 90th percentile | 10.918 px | 79.822 px |
| Maximum error | 419.013 px | 412.991 px |
| Within 5 px | 60.16% | 8.41% |
| Within 10 px | 87.81% | 24.22% |
| Within 20 px | 97.14% | 50.85% |

### 6-DoF pose from YOLO predictions

PnP returned **1,000/1,000** estimates in the 10K experiment and **99/100** in the smoke test. Errors below apply to successful estimates.

| Error statistic | 10K translation (m) | 10K rotation (°) | 1K translation (m) | 1K rotation (°) |
| --- | ---: | ---: | ---: | ---: |
| Mean | 0.6358 | 4.6462 | 5.9937 | 36.5437 |
| Median | 0.2326 | 2.3508 | 1.6720 | 16.1845 |
| 90th percentile | 0.6146 | 5.5193 | 15.0623 | 134.8385 |
| Maximum | 253.8642 | 179.5848 | 77.7202 | 176.3775 |

Both mean and median pose errors were lower in the 10K experiment. Extreme errors persist even when the solver returns a pose. In the 10K run, the largest translation error exceeded 250 m and the largest orientation error was nearly a half-turn.

### Geometric consistency check

Replacing YOLO predictions with ground-truth 2D projections checks the landmark correspondences, camera intrinsics, coordinate conventions, and solver together. Because this check bypasses the detector, it cannot measure the accuracy of the complete navigation pipeline. It succeeded on **1,000/1,000** images in the 10K experiment. In the smoke test it succeeded on **99/99** evaluated images, with mean errors of **0.000023 m** and **0.010814°**.

| 10K statistic with ground-truth keypoints | Translation error (m) | Rotation error (°) |
| --- | ---: | ---: |
| Mean | 0.000023 | 0.009438 |
| Median | 0.000020 | 0.000000 |
| 90th percentile | 0.000043 | 0.026846 |
| Maximum | 0.000068 | 0.040213 |

## Repository structure

| Stage | Initial 1K | Latest 10K |
| --- | --- | --- |
| Preprocessing | [Python script](src/prepare_hst_yolo.py) | [Colab preprocessing and ZIP creation](notebooks/prepare_hst_yolo_10k.ipynb) using the same script |
| YOLO training | [Training notebook](notebooks/yolo_training.ipynb) | [10K training notebook](notebooks/yolo_training_10k_HST.ipynb) |
| YOLO test and pixel errors | [Test notebook](notebooks/yolo_test_evaluation.ipynb) | [10K test notebook](notebooks/yolo_test_evaluation_10k.ipynb) |
| PnP evaluation | [Pose notebook](notebooks/pnp_evaluation.ipynb) | [10K pose notebook](notebooks/pnp_evaluation_10k.ipynb) |

Both pose notebooks import [the iterative solver](src/pnp.py). The [project metadata](pyproject.toml) specifies Python 3.10 or newer. The repository reserves `configs/`, `data/`, `docs/`, `results/`, `scripts/`, and `tests/` for configurations, local data, documentation, outputs, standalone scripts, and tests respectively. Dataset files and trained checkpoints are stored externally.

## Reproducibility

Run the preprocessing commands from the repository root. The script requires Pillow and a new or empty output directory. The notebooks contain their Colab setup cells. Mount Google Drive and select a GPU runtime for training and evaluation.

**1K smoke test.** Supply one image folder and its matching metadata file. Substitute the local input paths below, then place the converted dataset at `MyDrive/hst_yolo_pose/` for the 1K notebooks. Those evaluation notebooks also read the original metadata there. The PnP notebook additionally requires `_HST_Keypoints.json` and `_CameraIntrinsicMatrix.json`. Run the 1K training, test, and pose notebooks in the order shown above.

```bash
python src/prepare_hst_yolo.py --images "path/to/0000000" --metadata "path/to/0000000_metadata.json" --output "hst_yolo_pose"
```

**10K experiment.** The Drive layout for the recorded run is

```text
MyDrive/HST/
├── DLVS3-HST/
├── hst_yolo_pose_10k/
├── hst_yolo_pose_10k.zip
└── HST_YOLO_Runs/
    └── hst_pose_10k_sigma_010/
        └── weights/best.pt
```

1. **Preprocess.** Use the command below in Colab. The root contains ten numbered image folders, their matching `*_metadata.json` files, and the two geometry JSON files. If using the preprocessing notebook, set its `SOURCE` and `OUTPUT` variables to the paths below before running the conversion cells. Its ZIP cell already uses the `MyDrive/HST/` layout.

   ```python
   !python src/prepare_hst_yolo.py \
       --root "/content/drive/MyDrive/HST/DLVS3-HST" \
       --output "/content/drive/MyDrive/HST/hst_yolo_pose_10k"
   ```

2. **Archive and train.** Run the ZIP cell to archive the processed folder, including its top-level `hst_yolo_pose_10k/` directory. The 10K training notebook copies the ZIP from Drive to Colab, extracts it into `/content/hst_yolo_pose_10k/`, and trains on local storage. The evaluation notebooks load the run’s `weights/best.pt` from Drive.
3. **Evaluate YOLO.** Run the 10K test notebook on the held-out `test` split. It reports box and pose metrics, then compares predicted pixel coordinates with the keypoints in the processed labels.
4. **Evaluate PnP.** Run the 10K pose notebook with the same test images and checkpoint. It reads the ten original metadata files, `_CameraIntrinsicMatrix.json`, and `_HST_Keypoints.json` from `MyDrive/HST/DLVS3-HST/`. It uses `filename_map.json` when preprocessing renamed colliding image stems, checks ground-truth reprojection, and evaluates both predicted and annotated keypoints.

## Current status

The 10K experiment has completed preprocessing, training, held-out YOLO testing, 2D localization analysis, and PnP pose evaluation. Typical localization and pose errors decreased substantially from the initial smoke test. Investigating the images with extreme translation and rotation errors remains the next step.
