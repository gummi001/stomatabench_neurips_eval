"""Download StomataBench data or weights from the Hugging Face Hub."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATA_REPO = "Sainath001/StomataBench"
WEIGHTS_REPO = "Sainath001/stomatabench_neurips_weights"

SETS = ["OD-Valid", "T-MAWP", "T-DCWP", "T-MSWP", "T-GSH"]
PATTERNS = {
    "eval": ["datasets/*/coco.json", "metadata/*.csv", "predictions/*/*.json"],
    "images": ["datasets/*/coco.json", "metadata/*.csv", "datasets/T-MSWP/images_2/*"]
              + [f"datasets/{s}/images/*" for s in SETS],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["eval", "images", "weights"],
                    help="eval: annotations, metadata, predictions; images: annotations, metadata, images of the evaluation sets; "
                         "weights: the 15 fine-tuned checkpoints")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    if a.what == "weights":
        out = a.out or ROOT / "data" / "weights"
        snapshot_download(WEIGHTS_REPO, revision=a.revision, local_dir=out)
    else:
        out = a.out or ROOT / "data" / "StomataBench"
        snapshot_download(DATA_REPO, repo_type="dataset", revision=a.revision,
                          local_dir=out, allow_patterns=PATTERNS[a.what])
    print(f"downloaded {a.what} to {out}")


if __name__ == "__main__":
    main()
