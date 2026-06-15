import argparse
import os
import subprocess
import sys


WEIGHTS = {
    "ablation": {
        "file_id": "1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob",
        "filename": "bytetrack_ablation.pth.tar",
    },
    "mot17_x": {
        "file_id": "1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5",
        "filename": "bytetrack_x_mot17.pth.tar",
    },
}


def make_parser():
    parser = argparse.ArgumentParser("Download official ByteTrack checkpoints")
    parser.add_argument(
        "--name",
        choices=sorted(WEIGHTS),
        default="ablation",
        help="checkpoint to download",
    )
    parser.add_argument("--output-dir", default="pretrained")
    return parser


def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])


def main():
    args = make_parser().parse_args()
    ensure_gdown()

    import gdown

    weight = WEIGHTS[args.name]
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, weight["filename"])
    url = "https://drive.google.com/uc?id={}".format(weight["file_id"])

    print("Downloading {} to {}".format(args.name, output_path))
    gdown.download(url, output_path, quiet=False)
    if not os.path.exists(output_path):
        raise RuntimeError("Download failed: {}".format(output_path))
    print("Saved {}".format(output_path))


if __name__ == "__main__":
    main()
