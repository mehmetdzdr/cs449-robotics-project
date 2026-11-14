"""Generate the synthetic YOLO dataset (no manual labelling).

    python scripts/generate_dataset.py --n 2000 --out data/boxes        (headless: prefix with xvfb-run -a)
    python scripts/generate_dataset.py --n 8 --out data/preview --preview   also writes images with boxes drawn
"""
import argparse
from pathlib import Path

import cv2

from conveyor_sort.perception.dataset import CLASSES, generate

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=2000, help="number of images")
parser.add_argument("--out", type=Path, default=Path("data/boxes"))
parser.add_argument("--val-fraction", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--preview", action="store_true", help="draw the labels into copies of the images (visual check)")
args = parser.parse_args()

generate(args.out, args.n, args.val_fraction, args.seed)
print(f"wrote {args.n} images to {args.out}")

if args.preview:
    colors = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "gray": (200, 200, 200)}
    (args.out / "preview").mkdir(exist_ok=True)
    for image_path in sorted(args.out.glob("images/*/*.png")):
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        label_path = args.out / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        for line in label_path.read_text().split("\n"):
            if line:
                cls, xc, yc, bw, bh = line.split()
                p0 = (round((float(xc) - float(bw) / 2) * w) - 2, round((float(yc) - float(bh) / 2) * h) - 2)
                p1 = (round((float(xc) + float(bw) / 2) * w) + 2, round((float(yc) + float(bh) / 2) * h) + 2)
                cv2.rectangle(image, p0, p1, colors[CLASSES[int(cls)]], 1)
                cv2.putText(image, CLASSES[int(cls)], (p0[0], p0[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imwrite(str(args.out / "preview" / image_path.name), image)
    print(f"preview images in {args.out / 'preview'}")
