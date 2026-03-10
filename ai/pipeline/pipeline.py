import torch
import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import segmentation_models_pytorch as smp

class GlaucomaPipeline:
    def __init__(self, yolo_path, unet_path, masks_dir=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.masks_dir = Path(masks_dir) if masks_dir else None
        
        # 1. Ładowanie Detektora (YOLO)
        print(f"[*] Ładowanie YOLO z {yolo_path}...")
        self.yolo = YOLO(yolo_path).to(self.device)
        
        # 2. Ładowanie Segmentatora (U-Net++)
        print(f"[*] Ładowanie U-Net++ z {unet_path}...")
        self.unet = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=2,
            activation=None
        ).to(self.device)
        
        checkpoint = torch.load(unet_path, map_location=self.device)
        self.unet.load_state_dict(checkpoint['model_state_dict'])
        self.unet.eval()
        
        # Stałe preprocessingowe
        self.img_size = 512
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, img_bgr):
        """Przygotowanie tensora (Float32) dla U-Net++."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (self.img_size, self.img_size))
        img_norm = img_res.astype(np.float32) / 255.0
        img_norm = (img_norm - self.mean) / self.std
        tensor = torch.from_numpy(np.transpose(img_norm, (2, 0, 1)))
        return tensor.unsqueeze(0).to(self.device).float()

    def _load_gt(self, image_path):
        """Wczytuje Ground Truth jeśli folder masek został podany."""
        if not self.masks_dir: return None
        mask_path = self.masks_dir / f"{Path(image_path).stem}.png"
        if not mask_path.exists(): return None
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        return np.stack([(mask == 1), (mask == 2)], axis=0).astype(np.float32)

    def calculate_cdr(self, disc_mask, cup_mask):
        d_area, c_area = disc_mask.sum(), cup_mask.sum()
        return np.sqrt(c_area / d_area) if d_area > 0 else 0.0

    def run(self, image_path, conf=0.5):
        full_img = cv2.imread(str(image_path))
        if full_img is None: return None

        # KROK 1: Detekcja YOLO
        results = self.yolo(full_img, conf=conf, verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy()

        # Fallback: Jeśli YOLO zawiedzie, weź całe zdjęcie
        if len(bboxes) == 0:
            print("[!] YOLO nic nie znalazło. Przełączanie w tryb Full-Image ROI.")
            h, w = full_img.shape[:2]
            bboxes = np.array([[0, 0, w, h]])

        # Zbieramy wyniki (obsługa batchingu dla wielu detekcji)
        all_masks = []
        all_crops = []
        
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            crop = full_img[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            input_tensor = self._preprocess(crop)
            with torch.no_grad():
                output = self.unet(input_tensor)
                mask = torch.sigmoid(output).cpu().numpy()[0]
                all_masks.append(mask)
                all_crops.append((x1, y1, x2, y2))

        # Przygotowanie danych do wizualizacji (bierzemy pierwszy/główny obiekt)
        main_mask = all_masks[0]
        cdr_val = self.calculate_cdr(main_mask[0] > 0.5, main_mask[1] > 0.5)
        gt_masks = self._load_gt(image_path)
        cdr_gt = self.calculate_cdr(gt_masks[0], gt_masks[1]) if gt_masks is not None else 0.0

        return full_img, all_crops, all_masks, cdr_val, gt_masks, cdr_gt

def save_diagnostic_plot(image, crops, masks, cdr, gt_masks, cdr_gt, save_path):
    """Generuje zaawansowany raport graficzny."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1. Oryginał z ramką YOLO
    axes[0, 0].imshow(img_rgb)
    for (x1, y1, x2, y2) in crops:
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='cyan', linewidth=2)
        axes[0, 0].add_patch(rect)
    axes[0, 0].set_title("Oryginał + Detekcja ROI")

    # 2. Overlay Predykcji
    x1, y1, x2, y2 = crops[0]
    roi = img_rgb[y1:y2, x1:x2]
    roi_res = cv2.resize(roi, (512, 512))
    overlay = roi_res.copy()
    overlay[masks[0][0] > 0.5] = overlay[masks[0][0] > 0.5] * 0.5 + np.array([0, 255, 0]) * 0.5
    overlay[masks[0][1] > 0.5] = overlay[masks[0][1] > 0.5] * 0.5 + np.array([255, 0, 0]) * 0.5
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(f"Predykcja (CDR: {cdr:.3f})")

    # 3. Maska binarna predykcji
    axes[1, 0].imshow(masks[0][0], cmap='Greens_r', alpha=0.5)
    axes[1, 0].imshow(masks[0][1], cmap='Reds_r', alpha=0.5)
    axes[1, 0].set_title("Struktury: Dysk (G) i Kielich (R)")

    # 4. Ground Truth Comparison
    if gt_masks is not None:
        axes[1, 1].imshow(gt_masks[0], cmap='Greens_r', alpha=0.5)
        axes[1, 1].imshow(gt_masks[1], cmap='Reds_r', alpha=0.5)
        axes[1, 1].set_title(f"Ground Truth (GT CDR: {cdr_gt:.3f})")

    else:
        axes[1, 1].text(0.5, 0.5, "Brak danych GT", ha='center')

    diagnosis = "Podejrzenie JASKRY" if cdr > 0.65 else "ZDROWE / Ryzyko niskie"
    fig.suptitle(f"Analiza Jaskry: {diagnosis}\nPlik: {save_path.name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Ścieżka do zdjęcia')
    parser.add_argument('--yolo', type=str, default='./models/best.pt')
    parser.add_argument('--unet', type=str, default='./models/unetpp_best.pth')
    parser.add_argument('--masks', type=str, default='./data/masks')
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    pipeline = GlaucomaPipeline(args.yolo, args.unet, args.masks)
    
    print(f"[*] Przetwarzanie: {args.image}")
    output = pipeline.run(args.image)
    
    if output:
        img, crops, masks, cdr, gt_m, cdr_gt = output
        save_path = Path(args.output) / f"report_{Path(args.image).stem}.png"
        save_diagnostic_plot(img, crops, masks, cdr, gt_m, cdr_gt, save_path)
        print(f"[+] Raport zapisany w: {save_path}")
        print(f"[+] Wynik CDR: {cdr:.4f}")

if __name__ == "__main__":
    main()
