# Spacecraft Pose Estimation

## Project overview

This project investigates autonomous spacecraft navigation using monocular vision and deep learning to estimate a target spacecraft's six-degree-of-freedom (6-DoF) pose: its relative position and orientation. The current implementation combines **YOLOv8s-Pose** for 2D keypoint detection with a geometric **Perspective-n-Point (PnP)** solver for camera-relative pose estimation.

The complete pipeline has passed an initial functional smoke test on a small Hubble Space Telescope (HST) dataset sample. The results below are preliminary and do not constitute a final baseline.

## Dataset

The project uses **DLVS3-HST**, a synthetic HST dataset with ground-truth 6-DoF poses and **37 predefined keypoints**. The implementation uses PNG images, JSON annotations, the corresponding fixed 3D keypoints, and camera intrinsics.

Official dataset: [DLVS3-HST on Zenodo](https://zenodo.org/records/15731476).

The current smoke test used **1,000 images**, split into **800 training**, **100 validation**, and **100 test** images. Dataset files and trained weights are stored externally and are not included in this repository.

## Methodology

1. **Preprocess the dataset.** Match PNG images to JSON annotations, normalize the 37 annotated 2D keypoints, and generate approximate bounding boxes from valid keypoints with 10% padding, a minimum padding of two pixels per axis, and clipping to image boundaries. Write YOLO pose labels with `kpt_shape: [37, 3]`. Invalid or out-of-frame keypoints receive visibility `0`; in-frame annotations receive `1`, which does not establish whether a point is occluded. Numerical round-trip checks verify that normalized labels retain the original pixel coordinates. Split the matched image-label pairs reproducibly using seed `42` and an 80/10/10 ratio.
2. **Train the keypoint detector.** Fine-tune YOLOv8s-Pose to predict the 37 HST keypoints. The training notebook uses `yolov8s-pose.pt`, 100 epochs, image size `640`, batch size `8`, seed `42`, and patience `25`, with horizontal/vertical flipping and mosaic augmentation disabled.
3. **Estimate the relative pose.** Use the predicted 2D pixel coordinates, their corresponding fixed 3D coordinates, and camera intrinsics in OpenCV's `SOLVEPNP_ITERATIVE` solver. The initial implementation uses all 37 correspondences, with **no RANSAC or per-keypoint confidence filtering**, and returns a rotation matrix and translation vector.
4. **Evaluate against ground truth.** Compare estimated poses with the annotated 6-DoF poses after accounting for the dataset/OpenCV coordinate conventions. Translation error is the Euclidean distance between translation vectors; rotation error is the angular difference between rotation matrices. Repeat PnP with ground-truth 2D keypoints to check the geometric pipeline independently of detection errors.

## Initial smoke-test results

These results describe the initial functional experiment, **not final baseline performance**. The executed notebooks retain the recorded outputs; the figures below have not been recomputed as part of this documentation update.

### YOLO test-set performance

Keypoint mAP uses a **custom OKS sigma of 0.10** for each of the 37 keypoints.

| Metric | Bounding Box | Keypoints |
| --- | ---: | ---: |
| Precision | 1.000 | 0.876 |
| Recall | 0.999 | 0.860 |
| mAP50 | 0.995 | 0.789 |
| mAP50–95 | 0.882 | 0.542 |

### 2D keypoint localization

Pixel errors are measured against finite, in-frame annotated keypoints on images with a detection; the recorded evaluation includes 99 images and 3,662 keypoints.

| Metric | Error |
| --- | ---: |
| Mean pixel error | 34.887 px |
| Median pixel error | 19.527 px |

### PnP pose accuracy

| Metric | YOLO-predicted 2D keypoints | Ground-truth 2D keypoints |
| --- | ---: | ---: |
| Successful estimates | 99/100 images | 99/99 evaluated images |
| Mean translation error | 5.9937 m | 0.000023 m |
| Median translation error | 1.6720 m | — |
| Mean rotation error | 36.5437° | 0.010814° |
| Median rotation error | 16.1845° | — |

The ground-truth keypoint experiment uses the same 99 images with successful YOLO-based pose estimates, rather than all 100 test images. Its near-zero errors verify the geometric pipeline and coordinate conventions for that evaluated subset. The substantially larger errors with predicted keypoints show how 2D localization errors affect final pose accuracy. A successful solver return alone does not imply an accurate pose.

## Repository structure

| Path | Current contents and purpose |
| --- | --- |
| `configs/` | Placeholder for experiment configurations. |
| `data/` | Placeholder for local dataset storage; dataset files are external. |
| `docs/` | Placeholder for research documentation. |
| `notebooks/yolo_training.ipynb` | Colab training workflow and recorded training outputs. |
| `notebooks/yolo_test_evaluation.ipynb` | Held-out YOLO test metrics and pixel localization error analysis. |
| `results/` | Placeholder for evaluation outputs. |
| `scripts/` | Placeholder for standalone training/evaluation scripts. |
| `src/prepare_hst_yolo.py` | PNG/JSON preprocessing, label verification, dataset splitting, and `data.yaml` generation. |
| `src/pnp.py` | Iterative 37-point PnP function, without inference or dataset loading. |
| `src/pnp_evaluation.ipynb` | Colab pose evaluation using predicted and ground-truth 2D keypoints. |
| `tests/` | Placeholder for automated tests. |
| `pyproject.toml` | Python package metadata; currently declares no runtime dependencies. |

Placeholder directories contain `.gitkeep` files. Training and evaluation currently run through the notebooks, including the PnP evaluation notebook under `src/`.

## Reproducibility

### 1. Prepare the environment and dataset

The project metadata specifies Python **3.10 or newer**. For the standalone preprocessing and PnP modules, install their imported dependencies:

```bash
python -m pip install pillow numpy opencv-python-headless
```

The Colab notebooks install Ultralytics and their other dependencies in their setup cells. They use a GPU (`device=0`) and mount Google Drive. The recorded training and YOLO test outputs used Ultralytics `8.4.208` and `8.4.211`, respectively; the notebook installation commands are not version-pinned.

Obtain the dataset from the official source and make the PNG images and JSON annotations available locally or in Colab. From the repository root, run the preprocessing script, replacing these example paths with your own:

```bash
python src/prepare_hst_yolo.py --images "path/to/0000000" --metadata "path/to/0000000_metadata.json" --output "path/to/hst_yolo_pose"
```

All three arguments are required. Native Windows paths and Linux/Colab paths are supported. The script creates `images/{train,val,test}`, `labels/{train,val,test}`, and `data.yaml`. It processes all matched frames supplied to it; reproducing the smoke-test split requires the corresponding 1,000-image sample, rather than the full dataset.

### 2. Train YOLOv8s-Pose

Open [the training notebook](notebooks/yolo_training.ipynb) in Google Colab, select a GPU runtime, and run its cells in order. It expects the converted dataset at `/content/drive/MyDrive/hst_yolo_pose` and copies it to `/content/hst_yolo_pose`. Adjust the path variables if your storage differs.

The notebook updates `data.yaml` for Colab and sets `kpt_oks_sigmas: [0.10, ...]` for all 37 keypoints. The configured training run writes its checkpoint to:

```text
/content/drive/MyDrive/HST_YOLO_Runs/hst_pose_sigma_010/weights/best.pt
```

Check the actual saved run directory before evaluation, particularly after repeated training runs.

### 3. Evaluate the held-out YOLO test set

Run [the YOLO test evaluation notebook](notebooks/yolo_test_evaluation.ipynb) in cell order after training. It loads the saved `best.pt`, evaluates `split="test"` at image size `640` and batch size `8`, and then computes pixel errors against the original JSON keypoint annotations.

The metrics cell searches for the converted dataset under `/content/hst_yolo_pose`, `/content/drive/MyDrive/hst_yolo_pose_fixed`, or `/content/drive/MyDrive/hst_yolo_pose`. The pixel-error cell separately expects `0000000_metadata.json` under `/content/drive/MyDrive/hst_yolo_pose` and uses `/content/hst_yolo_pose/images/test`, falling back to that Drive dataset's `images/test`. Keep these paths consistent with your prepared data and checkpoint. Preprocessing does not copy the original metadata into the output directory.

### 4. Evaluate PnP poses

Run [the PnP evaluation notebook](src/pnp_evaluation.ipynb) in cell order with the trained checkpoint and test images available. Its setup cell clones this repository if necessary and imports `solve_pnp` from `src/pnp.py`.

The notebook expects the following original dataset files under `/content/drive/MyDrive/hst_yolo_pose`:

- `0000000_metadata.json`: annotated 2D keypoints and ground-truth transforms.
- `_HST_Keypoints.json`: the 37 corresponding fixed 3D keypoints.
- `_CameraIntrinsicMatrix.json`: the 3 × 3 camera intrinsic matrix.

It uses the same checkpoint path as YOLO test evaluation and looks for test images in `/content/hst_yolo_pose/images/test`, falling back to the Drive dataset's `images/test`. Adjust these variables for your environment. The notebook estimates poses, checks ground-truth reprojection, reports translation/rotation errors, and repeats PnP with ground-truth keypoints on the successful predicted-pose subset. `src/pnp.py` itself is an importable solver module, not an evaluation command.

## Current status

Preprocessing, keypoint training, held-out testing, and geometric pose evaluation have passed the initial functional smoke test. Future experiments will use a larger portion of DLVS3-HST and investigate improvements in keypoint localization and final pose accuracy. The current measurements remain preliminary; a final baseline has not yet been established.
