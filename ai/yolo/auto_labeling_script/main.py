import os
import cv2
import numpy as np


def preprocess(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def edge_map(img_gray):
    return cv2.Canny(img_gray, 30, 120)


def yolo_from_bbox(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    return x_center, y_center, width, height


def save_yolo_label(txt_path, class_id, x, y, w, h, img_w, img_h):
    x_center, y_center, width, height = yolo_from_bbox(x, y, w, h, img_w, img_h)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {x_center:.10f} {y_center:.10f} {width:.10f} {height:.10f}\n")


def draw_preview(image, x, y, w, h, score, out_path):
    vis = image.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        vis,
        f"score={score:.4f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(out_path, vis)


def find_best_square_roi(full_img, cropped_img, min_size=40, max_size=320, step=4):

    full_gray = preprocess(full_img)
    crop_gray = preprocess(cropped_img)

    full_edges = edge_map(full_gray)

    H, W = full_gray.shape[:2]
    best_score = -1.0
    best_box = None

    max_size = min(max_size, H, W)

    for size in range(min_size, max_size + 1, step):
        resized_crop_gray = cv2.resize(crop_gray, (size, size), interpolation=cv2.INTER_AREA)
        resized_crop_edges = edge_map(resized_crop_gray)

        result_gray = cv2.matchTemplate(full_gray, resized_crop_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val_gray, _, max_loc_gray = cv2.minMaxLoc(result_gray)


        result_edge = cv2.matchTemplate(full_edges, resized_crop_edges, cv2.TM_CCOEFF_NORMED)
        _, max_val_edge, _, max_loc_edge = cv2.minMaxLoc(result_edge)


        if max_val_gray >= max_val_edge:
            x, y = max_loc_gray
        else:
            x, y = max_loc_edge

        score = 0.7 * max_val_gray + 0.3 * max_val_edge

        if score > best_score:
            best_score = score
            best_box = (x, y, size, size, score)

    return best_box


def process_one_pair(full_image_path, cropped_image_path, output_txt_path,
                     class_id=0, preview_path=None,
                     min_size=40, max_size=320, step=4, min_score=0.45):
    full_img = cv2.imread(full_image_path)
    cropped_img = cv2.imread(cropped_image_path)

    if full_img is None:
        raise ValueError(f"Nie udało się wczytać obrazu: {full_image_path}")
    if cropped_img is None:
        raise ValueError(f"Nie udało się wczytać cropa: {cropped_image_path}")

    if full_img.shape[:2] != (512, 512):
        print(f"Uwaga: pełny obraz ma rozmiar {full_img.shape[:2]}, a nie (512, 512)")
    if cropped_img.shape[:2] != (640, 640):
        print(f"Uwaga: crop ma rozmiar {cropped_img.shape[:2]}, a nie (640, 640)")

    best = find_best_square_roi(
        full_img,
        cropped_img,
        min_size=min_size,
        max_size=max_size,
        step=step
    )

    if best is None:
        raise RuntimeError("Nie znaleziono dopasowania.")

    x, y, w, h, score = best

    if score < min_score:
        print(f"Uwaga: niski score dopasowania: {score:.4f}")

    img_h, img_w = full_img.shape[:2]
    save_yolo_label(output_txt_path, class_id, x, y, w, h, img_w, img_h)

    if preview_path is not None:
        draw_preview(full_img, x, y, w, h, score, preview_path)

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "score": score,
        "txt_path": output_txt_path,
    }


def process_folder(images_dir, cropped_dir, labels_dir, preview_dir=None,
                   class_id=0, min_size=40, max_size=320, step=4, min_score=0.45):
    os.makedirs(labels_dir, exist_ok=True)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(images_dir) if f.lower().endswith(exts)]

    ok_count = 0
    fail_count = 0

    for filename in files:
        name, _ = os.path.splitext(filename)
        full_path = os.path.join(images_dir, filename)

        crop_path = None
        for ext in exts:
            candidate = os.path.join(cropped_dir, name + ext)
            if os.path.exists(candidate):
                crop_path = candidate
                break

        if crop_path is None:
            print(f"[BRAK CROPA] {filename}")
            fail_count += 1
            continue

        txt_path = os.path.join(labels_dir, name + ".txt")
        preview_path = os.path.join(preview_dir, name + ".jpg") if preview_dir else None

        try:
            result = process_one_pair(
                full_image_path=full_path,
                cropped_image_path=crop_path,
                output_txt_path=txt_path,
                class_id=class_id,
                preview_path=preview_path,
                min_size=min_size,
                max_size=max_size,
                step=step,
                min_score=min_score
            )
            print(f"[OK] {filename} -> score={result['score']:.4f}")
            ok_count += 1
        except Exception as e:
            print(f"[BŁĄD] {filename} -> {e}")
            fail_count += 1

    print("\nPodsumowanie:")
    print(f"OK: {ok_count}")
    print(f"Błędy: {fail_count}")


if __name__ == "__main__":
    images_dir = "images"
    cropped_dir = "images_cropped"
    labels_dir = "labels"
    preview_dir = "preview"

    process_folder(
        images_dir=images_dir,
        cropped_dir=cropped_dir,
        labels_dir=labels_dir,
        preview_dir=preview_dir,
        class_id=0,
        min_size=40,
        max_size=320,
        step=4,
        min_score=0.45
    )

