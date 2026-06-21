# LTC Motion Predictor for ByteTrack

This is the simple path for Direction A: train a small continuous-time LTC/CfC residual predictor, then run ByteTrack with `--ltc_motion_ckpt`.

## 1. Prepare MOT17

Put MOT17 in this layout:

```text
datasets/mot/train/MOT17-02-FRCNN/gt/gt.txt
datasets/mot/train/MOT17-02-FRCNN/img1/*.jpg
...
```

On Kaggle, copy or symlink your MOT17 dataset into `datasets/mot/train`.

## 2. Train LTC Motion

```bash
python tools/train_ltc_motion.py \
  --data-root datasets/mot/train \
  --output outputs/ltc_motion_mot17.pth \
  --epochs 30 \
  --batch-size 512 \
  --hidden-size 128 \
  --num-layers 2 \
  --target-normalization standard \
  --split-mode half \
  --data-parallel
```

If memory is low, remove `--data-parallel` and use `--batch-size 256`.

## 3. Run ByteTrack With LTC

Use your normal detector checkpoint, and add the LTC checkpoint:

```bash
python tools/track.py \
  -f exps/example/mot/yolox_x_ablation.py \
  -c pretrained/bytetrack_ablation.pth.tar \
  -b 1 -d 1 --fp16 --fuse \
  --ltc_motion_ckpt outputs/ltc_motion_mot17.pth \
  --ltc_min_history 8 \
  --ltc_covariance_scale 1.0
```

Baseline is the same command without `--ltc_motion_ckpt`.

## 4. What To Compare

Run baseline ByteTrack first, then LTC ByteTrack on the same split/checkpoint.
Compare MOTA, IDF1, HOTA, ID switches, and mostly occlusion-heavy sequences.

The detector does not need to be retrained for this experiment. The LTC model only learns motion residuals from MOT ground truth.
