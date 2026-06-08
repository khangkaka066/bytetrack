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
!git clone https://github.com/<your-user>/<your-repo>.git
%cd <your-repo>
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

## Training Notes

Training requires the MOT/CrowdHuman/Cityperson/ETHZ data layout described in
`README.md`. Put large datasets and checkpoints in Kaggle Datasets or
`/kaggle/input`; do not commit them to GitHub. The existing `.gitignore` already
ignores `pretrained`, `YOLOX_outputs`, `*.pth`, `*.onnx`, and similar heavy
outputs.
