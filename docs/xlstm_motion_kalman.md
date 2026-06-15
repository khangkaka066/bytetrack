# Train va su dung ByteTrack + xLSTM Motion

Tai lieu nay mo ta cach train model xLSTM residual va cach bat ByteTrack da gan
xLSTM trong `yolox/tracker`.

Mac dinh ByteTrack van chay Kalman goc. xLSTM chi duoc dung khi truyen
`--xlstm_motion_ckpt`.

## Tong quan

ByteTrack goc dung Kalman state:

```text
[cx, cy, a, h, vx, vy, va, vh]
```

Trong do:

```text
cx, cy = tam bbox
a      = w / h
h      = chieu cao bbox
vx..vh = van toc cua 4 bien tren
```

xLSTM khong thay Kalman. No chi hoc phan sai so con lai sau Kalman prediction:

```text
residual = [dcx, dcy, da, dh]
target   = bbox_gt_xyah_t - bbox_kalman_xyah_t
```

Pipeline inference:

```text
1. Kalman predict:
   mean_kf, covariance_kf = Kalman.predict(...)

2. Lay motion history H cua track:
   H shape = [T, 12]

3. xLSTM predict:
   residual, log_var = xLSTM(H)

4. Sua Kalman mean:
   mean[:4] = mean[:4] + residual

5. Tang Kalman covariance theo uncertainty:
   diag(covariance[:4, :4]) += exp(log_var) * xlstm_covariance_scale

6. ByteTrack dung prediction moi de matching voi detection.
```

`log_var` duoc dung de bieu dien muc do khong chac chan cua residual. Khi model
khong chac, covariance lon hon, matching se bot tin vao prediction hon.

## File da tich hop

Code chinh:

```text
yolox/tracker/xlstm_motion.py
yolox/tracker/byte_tracker.py
```

CLI da them flag:

```text
tools/track.py
tools/demo_track.py
```

Chi tich hop ban chinh `yolox/tracker`. Cac ban `tutorials/`, `deploy/`,
`TensorRT`, `DeepStream` chua duoc gan xLSTM.

## Cai dat

Cai dependency ByteTrack nhu binh thuong, sau do cai xLSTM:

```bash
pip install xlstm
```

Neu dung CUDA sLSTM backend cua NX-AI/xlstm, nen dung GPU NVIDIA co CUDA phu hop.
Neu gap loi kernel tren may khong ho tro CUDA backend, thu chay voi:

```bash
--xlstm_backend vanilla
--xlstm_device cpu
```

Tuy nhien, khi train/inference that su, nen uu tien GPU.

## Motion history

Moi timestep trong history co 12 chieu:

```text
[
  cx,
  cy,
  a,
  h,
  vx,
  vy,
  va,
  vh,
  delta_t,
  is_missing,
  missing_count,
  confidence
]
```

Khuyen nghi:

```text
history_len = 16
input_dim   = 12
```

Quy tac ghi history:

```text
Track duoc match detection:
  is_missing = 0
  missing_count = 0
  confidence = detection score

Track bi lost / khong match:
  is_missing = 1
  missing_count += 1
  confidence = 0.0
```

## Chuan bi du lieu train

Dung ground truth MOT format, vi du:

```text
datasets/mot/train/MOT17-02-FRCNN/gt/gt.txt
datasets/mot/train/MOT17-04-FRCNN/gt/gt.txt
...
```

MOT `gt.txt` thuong co format:

```text
frame,id,x,y,w,h,mark,class,visibility
```

Khi train motion model:

```text
1. Chi dung object co mark = 1.
2. Thuong chi dung pedestrian class = 1 voi MOT17/MOT20.
3. Group theo sequence va track id.
4. Sort moi track theo frame tang dan.
5. Convert tlwh -> xyah:
   cx = x + w / 2
   cy = y + h / 2
   a  = w / h
   h  = h
```

## Cach tao sample train

Train sample tai frame `t` phai chi dung history truoc frame `t`.

Pseudo-code:

```python
from collections import deque
import numpy as np

from yolox.tracker.kalman_filter import KalmanFilter


def tlwh_to_xyah(tlwh):
    x, y, w, h = tlwh
    return np.asarray([x + w / 2, y + h / 2, w / h, h], dtype=np.float32)


def make_history_feature(mean, frame_delta, is_missing, missing_count, confidence):
    return np.asarray([
        mean[0], mean[1], mean[2], mean[3],
        mean[4], mean[5], mean[6], mean[7],
        frame_delta,
        1.0 if is_missing else 0.0,
        float(missing_count),
        float(confidence),
    ], dtype=np.float32)


def build_samples(track_rows, history_len=16):
    kf = KalmanFilter()
    history = deque(maxlen=history_len)
    samples = []

    first = track_rows[0]
    mean, covariance = kf.initiate(tlwh_to_xyah(first["tlwh"]))
    history.append(make_history_feature(
        mean=mean,
        frame_delta=1.0,
        is_missing=False,
        missing_count=0,
        confidence=first.get("score", 1.0),
    ))

    last_frame = first["frame"]
    missing_count = 0

    for row in track_rows[1:]:
        frame_delta = max(1, row["frame"] - last_frame)

        mean_pred, cov_pred = kf.predict(mean.copy(), covariance.copy())
        gt_xyah = tlwh_to_xyah(row["tlwh"])
        target_residual = gt_xyah - mean_pred[:4]

        if len(history) == history_len:
            samples.append({
                "history": np.stack(history).astype(np.float32),
                "target_residual": target_residual.astype(np.float32),
            })

        mean, covariance = kf.update(mean_pred, cov_pred, gt_xyah)
        missing_count = 0
        history.append(make_history_feature(
            mean=mean,
            frame_delta=frame_delta,
            is_missing=False,
            missing_count=missing_count,
            confidence=row.get("score", 1.0),
        ))
        last_frame = row["frame"]

    return samples
```

Neu mot object bi mat frame giua chung, co the chen cac missing step bang
Kalman prediction truoc khi gap lai ground truth. Khi chen missing step:

```text
mean, covariance = kf.predict(mean, covariance)
missing_count += 1
append history voi is_missing = 1, confidence = 0.0
```

Ban train tot nhat nen match dung cach inference sinh history trong
`STrack._append_motion_history`.

## Model train

Dung dung class da co:

```python
from yolox.tracker.xlstm_motion import XlstmMotionResidual

model = XlstmMotionResidual(
    input_dim=12,
    history_len=16,
    embedding_dim=128,
    num_blocks=4,
    num_heads=4,
    backend="cuda",
).cuda()
```

Input/output:

```text
history:         [batch, 16, 12]
pred_residual:   [batch, 4]
pred_log_var:    [batch, 4]
target_residual: [batch, 4]
```

## Loss train

Dung heteroscedastic Gaussian loss:

```python
import torch


def residual_nll_loss(pred_residual, pred_log_var, target_residual):
    pred_log_var = torch.clamp(pred_log_var, min=-10.0, max=10.0)
    diff = target_residual - pred_residual
    loss = 0.5 * torch.exp(-pred_log_var) * diff.pow(2) + 0.5 * pred_log_var
    loss = loss.mean()
    loss = loss + 1e-4 * pred_log_var.pow(2).mean()
    return loss
```

Ly do dung loss nay:

```text
residual hoc sua sai so Kalman
log_var hoc uncertainty cua sai so do
inference dung exp(log_var) de tang covariance
```

## Vong train mau

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from yolox.tracker.xlstm_motion import XlstmMotionResidual


history_tensor = torch.from_numpy(histories).float()
target_tensor = torch.from_numpy(target_residuals).float()
dataset = TensorDataset(history_tensor, target_tensor)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

model = XlstmMotionResidual(
    input_dim=12,
    history_len=16,
    embedding_dim=128,
    num_blocks=4,
    num_heads=4,
    backend="cuda",
).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

for epoch in range(50):
    model.train()
    total_loss = 0.0

    for history, target in loader:
        history = history.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        pred_residual, pred_log_var = model(history)
        loss = residual_nll_loss(pred_residual, pred_log_var, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * history.size(0)

    print("epoch", epoch, "loss", total_loss / len(dataset))

torch.save({"model": model.state_dict()}, "outputs/xlstm_motion.pth")
```

## Luu checkpoint

Inference wrapper chap nhan 2 format:

```python
torch.save(model.state_dict(), "outputs/xlstm_motion.pth")
```

hoac:

```python
torch.save({"model": model.state_dict()}, "outputs/xlstm_motion.pth")
```

Nen dung format thu hai de sau nay them metadata:

```python
torch.save({
    "model": model.state_dict(),
    "history_len": 16,
    "input_dim": 12,
    "embedding_dim": 128,
    "num_blocks": 4,
    "num_heads": 4,
}, "outputs/xlstm_motion.pth")
```

Luu y: khi inference, cac flag architecture phai khop voi checkpoint:

```text
--xlstm_history_len
--xlstm_input_dim
--xlstm_embedding_dim
--xlstm_num_blocks
--xlstm_num_heads
```

## Su dung voi tools/track.py

Chay ByteTrack goc, khong xLSTM:

```bash
python tools/track.py \
  -f exps/example/mot/yolox_x_ablation.py \
  -c pretrained/bytetrack_x_mot17.pth.tar
```

Chay ByteTrack + xLSTM:

```bash
python tools/track.py \
  -f exps/example/mot/yolox_x_ablation.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  --xlstm_motion_ckpt outputs/xlstm_motion.pth \
  --xlstm_history_len 16 \
  --xlstm_min_history 16 \
  --xlstm_embedding_dim 128 \
  --xlstm_num_blocks 4 \
  --xlstm_num_heads 4 \
  --xlstm_covariance_scale 1.0
```

Neu checkpoint train voi cau hinh khac, doi flags cho khop.

## Su dung voi tools/demo_track.py

Video demo:

```bash
python tools/demo_track.py video \
  -f exps/example/mot/yolox_x_ablation.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  --path videos/palace.mp4 \
  --save_result \
  --xlstm_motion_ckpt outputs/xlstm_motion.pth \
  --xlstm_history_len 16 \
  --xlstm_min_history 16
```

Webcam:

```bash
python tools/demo_track.py webcam \
  -f exps/example/mot/yolox_x_ablation.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  --camid 0 \
  --xlstm_motion_ckpt outputs/xlstm_motion.pth
```

## Cac flag xLSTM

```text
--xlstm_motion_ckpt
  Path checkpoint xLSTM. Neu khong co flag nay thi tat xLSTM.

--xlstm_history_len
  So timestep history T. Mac dinh 16.

--xlstm_input_dim
  So chieu feature moi timestep. Mac dinh 12.

--xlstm_min_history
  So timestep toi thieu truoc khi bat xLSTM cho mot track.
  Nen de bang history_len khi eval nghiem tuc.

--xlstm_embedding_dim
  Hidden size cua xLSTM backbone.

--xlstm_num_blocks
  So block xLSTM.

--xlstm_num_heads
  So head trong xLSTM config.

--xlstm_backend
  Backend sLSTM. Mac dinh cuda.

--xlstm_device
  Device load xLSTM, vi du cuda, cuda:0, cpu.

--xlstm_covariance_scale
  He so nhan exp(log_var) khi cong vao covariance.
  Dat 0.0 de test residual-only.

--xlstm_max_abs_residual
  Clip residual de tranh checkpoint xau lam bbox nhay qua lon.
```

## Ablation nen chay

Nen so sanh it nhat 3 che do:

```text
1. Kalman only:
   khong truyen --xlstm_motion_ckpt

2. Kalman + xLSTM residual only:
   --xlstm_motion_ckpt outputs/xlstm_motion.pth
   --xlstm_covariance_scale 0.0

3. Kalman + xLSTM residual + uncertainty:
   --xlstm_motion_ckpt outputs/xlstm_motion.pth
   --xlstm_covariance_scale 1.0
```

Neu mode 2 tot hon mode 1 nhung mode 3 te hon, log_var dang scale chua tot.
Thu `--xlstm_covariance_scale 0.1`, `0.25`, `0.5`, `1.0`.

## Debug nhanh

Neu ket qua te hon Kalman goc:

```text
1. Kiem tra target residual co dung xyah khong, khong phai tlwh.
2. Kiem tra history train va history inference cung format.
3. Kiem tra checkpoint architecture khop voi CLI flags.
4. Giam --xlstm_max_abs_residual xuong 64 hoac 128.
5. Thu --xlstm_covariance_scale 0.0 de tach loi residual va log_var.
6. Kiem tra residual da duoc normalize khi train hay chua.
```

Neu train target da normalize, inference cung phai unnormalize residual truoc khi
cong vao `mean[:4]`. Code hien tai gia dinh residual o don vi Kalman goc, tuc la
khong normalize output.

## Khuyen nghi train

Bat dau voi:

```text
history_len = 16
embedding_dim = 128
num_blocks = 4
num_heads = 4
lr = 1e-4
batch_size = 256
epochs = 30-50
```

Nen chia validation theo sequence, khong random tung sample, de do kha nang
generalize sang video moi.

Metric nen xem:

```text
validation residual MAE
validation NLL
MOTA / IDF1 / HOTA sau khi chay tools/track.py
so ID switches
```

Dung MOT metric moi quyet dinh checkpoint cuoi, vi residual loss thap chua chac
tracking da tot.
