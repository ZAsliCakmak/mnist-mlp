
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from data import MNIST_MEAN, MNIST_STD
from model import MLP


def load_gray(path):
   
    try:
        pil = ImageOps.exif_transpose(Image.open(path)).convert("L")
    except Exception as e:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}") from e
    img = np.array(pil)
    
    if img.shape[0] > 600:
        scale = 600 / img.shape[0]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def binarize(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def segment_digits(binary):
    """Rakamların sınır kutularını soldan sağa sıralı döndürür."""
   
    dilated = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours]
    if not boxes:
        return []
    max_h = max(h for _, _, _, h in boxes)
    
    boxes = [b for b in boxes if b[3] >= 0.4 * max_h]

    
    median_h = float(np.median([b[3] for b in boxes]))
    boxes = sorted(boxes, key=lambda b: b[1] + b[3] / 2)
    rows, current = [], [boxes[0]]
    for b in boxes[1:]:
        prev_cy = np.mean([p[1] + p[3] / 2 for p in current])
        if abs((b[1] + b[3] / 2) - prev_cy) > 0.6 * median_h:
            rows.append(current)
            current = [b]
        else:
            current.append(b)
    rows.append(current)

    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[0]))
    return ordered


def to_mnist_format(digit):
    """Siyah zemin / beyaz rakam olan kırpılmış görüntüyü 28x28 MNIST formatına çevirir."""
    h, w = digit.shape

    
    k = max(1, int(round(max(h, w) * 0.04)))
    digit = cv2.dilate(digit, np.ones((k, k), np.uint8), iterations=1)

    
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y0, x0 = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = small

    
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        shift = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        canvas = cv2.warpAffine(canvas, shift, (28, 28), borderValue=0)
    return canvas


def normalize(img28):
    x = img28.astype(np.float32) / 255.0
    x = (x - MNIST_MEAN) / MNIST_STD
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Okul numarasının fotoğrafı")
    parser.add_argument("--number", default=None, help="Gerçek okul numarası (karşılaştırma için)")
    parser.add_argument("--model", default="results/mlp_model.pth")
    args = parser.parse_args()

    out_dir = "results/own_digits"
    os.makedirs(out_dir, exist_ok=True)

    model = MLP()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    gray = load_gray(args.image)
    binary = binarize(gray)
    boxes = segment_digits(binary)
    print(f"Bulunan rakam sayısı: {len(boxes)}")

    if args.number and len(boxes) != len(args.number):
        print(f"UYARI: Numara {len(args.number)} haneli ama {len(boxes)} rakam bulundu. "
              "Fotoğrafı sadece numarayı içerecek şekilde kırp, rakamlar arasında boşluk bırak.")

    crops, processed, preds, confs = [], [], [], []
    for i, (x, y, w, h) in enumerate(boxes):
        digit = binary[y:y + h, x:x + w]
        img28 = to_mnist_format(digit)

        with torch.no_grad():
            probs = torch.softmax(model(normalize(img28)), dim=1)[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])

        crops.append(gray[y:y + h, x:x + w])
        processed.append(img28)
        preds.append(pred)
        confs.append(conf)
        plt.imsave(os.path.join(out_dir, f"digit_{i + 1}_pred{pred}.png"), img28, cmap="gray")

    predicted_number = "".join(str(p) for p in preds)

    print("\nSıra | Tahmin | Güven  | Gerçek | Doğru?")
    print("-----+--------+--------+--------+-------")
    correct = 0
    for i, (p, c) in enumerate(zip(preds, confs)):
        true = args.number[i] if args.number and i < len(args.number) else "-"
        ok = "-" if true == "-" else ("✓" if str(p) == true else "✗")
        correct += ok == "✓"
        print(f"{i + 1:4d} | {p:6d} | {c * 100:5.1f}% | {true:>6} | {ok:>5}")

    print(f"\nTahmin edilen numara : {predicted_number}")
    if args.number:
        n = min(len(preds), len(args.number))
        print(f"Gerçek numara        : {args.number}")
        print(f"Rakam bazında doğruluk: {correct}/{len(args.number)} "
              f"= {correct / len(args.number) * 100:.1f}%")

    
    n = len(boxes)
    if n:
        fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.8), squeeze=False)
        for i in range(n):
            axes[0, i].imshow(crops[i], cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].imshow(processed[i], cmap="gray")
            axes[1, i].axis("off")
            title = f"{preds[i]} ({confs[i] * 100:.0f}%)"
            color = "black"
            if args.number and i < len(args.number):
                color = "green" if str(preds[i]) == args.number[i] else "red"
            axes[1, i].set_title(title, color=color, fontsize=10)
        axes[0, 0].set_ylabel("Orijinal")
        fig.suptitle(f"Tahmin: {predicted_number}" + (f"  |  Gerçek: {args.number}" if args.number else ""))
        plt.tight_layout()
        plt.savefig("results/own_predictions.png", dpi=120)
        print("\nKaydedildi: results/own_predictions.png ve results/own_digits/")


if __name__ == "__main__":
    main()
