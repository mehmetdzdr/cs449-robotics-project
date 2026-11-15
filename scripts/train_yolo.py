"""Fine-tune a small pretrained YOLO model on the synthetic dataset (needs a GPU and `pip install -e ".[learning]"`).

    python scripts/generate_dataset.py --n 2000 --out data/boxes
    python scripts/train_yolo.py --data data/boxes/data.yaml --epochs 30

The best weights are copied to weights/yolo_boxes.pt, where `YoloDetector` looks for them.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from conveyor_sort.perception.yolo import DEFAULT_WEIGHTS, IMAGE_SIZE

if __name__ == "__main__":       # the data loader's worker processes import this file again
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/boxes/data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt", help="pretrained model to start from (downloaded by Ultralytics)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="directory for training logs and checkpoints")
    parser.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    data, runs = str(args.data.resolve()), str(args.runs.resolve())

    model = YOLO(args.model)
    model.train(data=data, epochs=args.epochs, imgsz=IMAGE_SIZE, batch=args.batch, rect=True,
                hsv_h=0.0,                       # no hue augmentation: the color is the class
                project=runs, name="yolo_boxes", exist_ok=True)
    best = args.runs / "yolo_boxes" / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=data, imgsz=IMAGE_SIZE, rect=True, project=runs, name="yolo_boxes_val",
                                  exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)
    print(f"validation: mAP50 {metrics.box.map50:.4f}, mAP50-95 {metrics.box.map:.4f}, "
          f"precision {metrics.box.mp:.4f}, recall {metrics.box.mr:.4f}")
    print(f"inference {metrics.speed['inference']:.1f} ms per image (validation batches)")
    print(f"weights copied to {args.out}")
