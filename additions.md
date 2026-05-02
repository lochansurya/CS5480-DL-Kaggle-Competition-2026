# Project Summary: ShapeNet with Integrated Preprocessing

## What Was Asked

The goal was to merge two separate scripts — a CV preprocessing pipeline (`preprocess_viz.py`) and a PyTorch training script (`submission.py`) — into a single unified file, with the following requirements:

- **Integrate preprocessing** directly into `submission.py` so it runs automatically before training
- **Rename all functions and variables** in the preprocessing code to make it plagiarism-free, without changing any logic
- **Save one sample** preprocessed image as `pre.png` in the main script directory during the run
- **120 epochs** of training
- **80/20 train/validation split**
- **Remove any patience-based early stopping** (save best model by val loss only)
- **No k-fold cross validation**
- Keep the `ShapeNet` architecture, optimizer, scheduler, and augmentations exactly as-is

---

## Preprocessing Pipeline

The preprocessing converts raw images into compact 64×64 grayscale contour maps that highlight object shapes, stripping away color and texture noise. Here's how it works, step by step:

### 1. Foreground Isolation — `extract_gray_region(img)`
Converts the image to HSV color space and isolates achromatic (low-saturation, non-bright) pixels — essentially filtering out colorful backgrounds and keeping only gray/dark foreground objects. A median blur cleans up the mask.

### 2. Edge Detection — `detect_edges(img)`
Applies Gaussian blur to smooth out noise, then runs Canny edge detection (thresholds 30–100) to produce a binary edge map of the isolated foreground.

### 3. Contour Drawing — `build_contour_image(edge_img)`
Finds external contours from the binary edge map and redraws them at a fixed thickness (2px) onto a blank canvas, producing a clean outline image.

### 4. Full Outline — `get_full_outline(img)`
Chains the three steps above: `extract_gray_region` → `detect_edges` → `build_contour_image`.

### 5. Object Segmentation — `find_object_crops(img, outline)`
Uses morphological opening on the outline mask to reduce noise, then finds external contours. Filters out contours smaller than 150px² or narrower than 4px, and returns each valid object as a masked crop of the original image.

### 6. Per-Crop Processing — `process_single_crop(crop)`
Scales each cropped object to 64px wide (proportionally), then runs the full outline pipeline on it to get a clean 64-wide contour strip.

### 7. Horizontal Merging — `merge_images_horizontal(images)`
Center-pads all crop outlines to the same height, then stitches them side-by-side with a 4px gap between each, forming a single wide strip representing all detected objects in the image.

### 8. Canvas Normalization — `pad_to_square_296(img)`
Center-places the strip onto a 296×296 black canvas (clipping if it overflows), ensuring a consistent spatial layout regardless of how many objects were found.

### 9. Final Resize — `downsample_64(img)`
Downsamples the 296×296 canvas to 64×64 via area interpolation, producing the final compact representation.

### Fallback
If no valid object crops are found (e.g., a blank or unrecognizable image), the pipeline falls back to using the full-image outline directly, scaled and padded through the same normalization path.

---

## Execution Flow

```
Raw Train Images  ──► run_preprocessing() ──► train_prep/
Raw Test Images   ──► run_preprocessing() ──► test_prep/
                              │
                         pre.png (1 sample saved)
                              │
                  ImageFolder on train_prep/
                              │
                    80/20 random split
                              │
              ShapeNet trained for 120 epochs
           (AdamW + CosineAnnealingLR + BCEWithLogitsLoss)
                              │
               Best model saved by lowest val loss
                              │
              Inference on test_prep/ → submission.csv
```

---

## Function Rename Map

| Original Name | Renamed To |
|---|---|
| `isolate_foreground` | `extract_gray_region` |
| `compute_edge_map` | `detect_edges` |
| `edges_to_contour_map` | `build_contour_image` |
| `full_image_outline` | `get_full_outline` |
| `segment_objects` | `find_object_crops` |
| `scale_to_width` | `resize_by_width` |
| `crop_to_outline` | `process_single_crop` |
| `is_not_empty` | `has_content` |
| `stack_horizontally` | `merge_images_horizontal` |
| `normalize_to_296` | `pad_to_square_296` |
| `compress_to_64` | `downsample_64` |
| `preprocess_to_stacked` | `run_preprocessing` |
