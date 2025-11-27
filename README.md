# ⭐ **BurstCleaner**
### _A hybrid time- and vision-based burst photo cleaner_

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Cross--Platform-lightgrey?style=for-the-badge">
  <img src="https://img.shields.io/badge/Deep%20Learning-ResNet18%20%2B%20ResNet50-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge">
</p>

BurstCleaner automatically detects **photo bursts** — sequences of similar images taken within short intervals — and selects the **best image** using hybrid visual similarity (ResNet18 + ResNet50).

Designed for:
- Cleaning messy camera folders  
- Removing duplicates & near-duplicates  
- Keeping only the best shot in each burst  
- Running efficiently on desktop systems  

---

## 📦 Features

### ✔ Time-based clustering  
Detect bursts using timestamp deltas (1D DBSCAN-like).

### ✔ Hybrid visual similarity  
Two neural networks:
- **ResNet50** → strict, detail-sensitive  
- **ResNet18** → flexible, general shape-based  

A burst is accepted only if both agree.

### ✔ Memory-efficient  
Embeddings are computed **only** inside bursts and cleared immediately.

### ✔ Cross-platform core  
Only loaders & embedders are platform-specific — all logic is OS-agnostic.

### ✔ JSON output  
Includes burst composition, similarity metrics, and recommended image.

---

## 📁 Project Structure

```
burst_cleaner/
├── README.md                # This file!
├── cli.py                   # CLI entrypoint
├── config.py                # Default settings
├── requirements.txt
└── burst_cleaner/
    ├── core/
    │   ├── loader_core.py       # Abstract loader interface
    │   ├── embeddings_core.py   # Abstract embedding backend
    │   ├── clustering.py        # Time-based burst detection
    │   ├── similarity.py        # Cosine similarity, centroid, best-image selection
    │   └── pipeline.py          # Full processing pipeline (hybrid strict)
    │
    └── platform_adapters/
        ├── windows_loader.py    # Windows implementation
        └── windows_embeddings.py# Torch embeddings (ResNet18/50)
```

---

## ⚙️ Installation

### 1. Create a virtual environment

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

Or manually:

```
pip install torch torchvision pillow numpy exifread
```

---

## 🚀 Usage (CLI)

From the project root:

```
python cli.py --input-folder "C:\Photos" --verbose
```

### Arguments

| Flag | Description |
|------|-------------|
| `--input-folder` | Folder containing images (required) |
| `--output-json` | Output JSON file (default: `bursts.json`) |
| `--time-gap-max` | Max seconds between images inside a burst |
| `--min-burst-len` | Minimum images required to form a burst |
| `--similarity-threshold` | Cosine threshold for both networks |
| `--verbose` | Print detailed burst info |

### Example

```bash
python cli.py `
  --input-folder "C:\Users\Me\Pictures" `
  --output-json "C:\Users\Me\Desktop\results.json" `
  --time-gap-max 3 `
  --min-burst-len 3 `
  --similarity-threshold 0.80 `
  --verbose
```

---

## 📄 Output (JSON)

```json
{
  "folder": "C:\\\\Pictures",
  "num_images": 27,
  "num_bursts": 2,
  "bursts": [
    {
      "burst_id": 1,
      "num_images": 7,
      "image_ids": [
        "C:\\\\...\\\\IMG_001.jpg",
        "C:\\\\...\\\\IMG_002.jpg"
      ],
      "avg_similarity_strict": 0.87,
      "avg_similarity_loose": 0.90,
      "recommended_keep": "C:\\\\...\\\\IMG_002.jpg"
    }
  ]
}
```

---

## 🧠 How the Hybrid System Works

**ResNet50**  
- High-dimensional embedding  
- Sensitive to texture & detail  

**ResNet18**  
- Lower-dimensional  
- Sensitive to structure & color pattern  

A burst is accepted only if:

```
sim50 >= threshold  AND  sim18 >= threshold
```

This prevents:
- **False merges** (images too different)
- **False splits** (similar but slightly varied shots)

---

## 📌 Design Goals

- Fast enough for folders with thousands of images  
- OS-independent core logic  
- No GPU required (but used automatically if available)  
- Human-friendly CLI  
- Clean extensible pipeline (Android, Linux loaders, etc.)

---

## 🛠 Roadmap

- [ ] Android-compatible loader  
- [ ] GUI for file management  
- [ ] Auto-move “recommended_keep” images into `keep/` folder  
- [ ] Auto-delete or archive leftovers  
- [ ] Expose tuning controls for Strict/Loose thresholds  
- [ ] GitHub Actions workflow for CI  

---

## 📜 License

MIT License
