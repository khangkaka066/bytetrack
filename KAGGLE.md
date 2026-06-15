# Run ByteTrack on Kaggle

This repository is based on the original ByteTrack codebase, with a small NumPy
2.x compatibility fix in the main tracker path.

## Push to your GitHub

The current `origin` remote points to the upstream ByteTrack repository. Point
it to your repository before pushing:

```bash
git remote rename origin upstream
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

If your GitHub repository already exists and uses SSH, use this instead:

```bash
git remote add origin git@github.com:<your-user>/<your-repo>.git
git push -u origin main
```

## Kaggle Notebook Setup

Create a Kaggle Notebook with GPU enabled, then run:

```python
!git clone https://github.com/khangkaka066/bytetrack.git
%cd bytetrack
!pip install -q -r requirements-kaggle.txt
```

For demo inference, download a ByteTrack checkpoint into `pretrained/`:

```python
!mkdir -p pretrained
# Put bytetrack_x_mot17.pth.tar in pretrained/
```

Then run the video demo:

```python
!python tools/demo_track.py video \
  -f exps/example/mot/yolox_x_mix_det.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  --device gpu \
  --fp16 \
  --fuse \
  --save_result
```

Outputs are written under `YOLOX_outputs/`.

## MOT17 Tracking

Use this section when you want to run ByteTrack on MOT17 instead of the small
demo video.

### 1. Prepare MOT17

Add MOT17 as a Kaggle Dataset first. The dataset should contain `train/` and
`test/` folders, for example:

```text
/kaggle/input/mot17/MOT17/train
/kaggle/input/mot17/MOT17/test
```

Copy MOT17 into the working directory because the conversion script writes
annotation files:

```python
!mkdir -p datasets
!cp -r /kaggle/input/mot17/MOT17 datasets/mot
!python tools/convert_mot17_to_coco.py
```

If your Kaggle Dataset exposes `train/` and `test/` directly, use this copy
command instead:

```python
!mkdir -p datasets/mot
!cp -r /kaggle/input/mot17/train datasets/mot/train
!cp -r /kaggle/input/mot17/test datasets/mot/test
!python tools/convert_mot17_to_coco.py
```

After conversion, these files should exist:

```python
!ls datasets/mot/annotations
```

### 2. Prepare Checkpoint

Create `pretrained/` and place the MOT17 checkpoint there:

```python
!mkdir -p pretrained
# Put bytetrack_x_mot17.pth.tar in pretrained/
```

If the checkpoint is also attached as a Kaggle Dataset:

```python
!cp /kaggle/input/bytetrack-weights/bytetrack_x_mot17.pth.tar pretrained/
```

Or download the official ByteTrack ablation checkpoint directly from Google
Drive when Kaggle internet is enabled:

```python
!python tools/download_bytetrack_weights.py --name ablation --output-dir pretrained
```

### 3. Run MOT17 Test Tracking

This runs on `datasets/mot/test` and writes result text files for MOTChallenge:

```python
!python tools/track.py \
  -f exps/example/mot/yolox_x_mix_det.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  -b 1 \
  -d 1 \
  --fp16 \
  --fuse
```

The MOT17 result files are written here:

```python
!ls YOLOX_outputs/yolox_x_mix_det/track_results
```

### 4. Run MOT17 Half-Val Evaluation

Use this first if you want to check that the pipeline works and get metrics in
the notebook. This needs the ablation checkpoint:

```python
!cp /kaggle/input/bytetrack-weights/bytetrack_ablation.pth.tar pretrained/

!python tools/track.py \
  -f exps/example/mot/yolox_x_ablation.py \
  -c pretrained/bytetrack_ablation.pth.tar \
  -b 1 \
  -d 1 \
  --fp16 \
  --fuse
```

The half-val result files are written here:

```python
!ls YOLOX_outputs/yolox_x_ablation/track_results
```

## Training Notes

Training requires the MOT/CrowdHuman/Cityperson/ETHZ data layout described in
`README.md`. Put large datasets and checkpoints in Kaggle Datasets or
`/kaggle/input`; do not commit them to GitHub. The existing `.gitignore` already
ignores `pretrained`, `YOLOX_outputs`, `*.pth`, `*.onnx`, and similar heavy
outputs.
