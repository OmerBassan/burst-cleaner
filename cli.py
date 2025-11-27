import argparse
import json
from .config import DEFAULT_CONFIG
from .platform_adapters.windows_loader import WindowsImageLoader
from .platform_adapters.windows_embeddings import DesktopTorchEmbeddingBackend
from .core.pipeline import pipeline_bursts_with_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="BurstCleaner CLI")

    # קלט / פלט
    parser.add_argument("--input-folder", required=True,
                        help="Path to folder with images")
    parser.add_argument("--output-json", default="bursts.json",
                        help="Where to save the output")

    # פרמטרים
    parser.add_argument("--time-gap-max", type=float,
                        default=DEFAULT_CONFIG["time_gap_max"])
    parser.add_argument("--min-burst-len", type=int,
                        default=DEFAULT_CONFIG["min_burst_len"])
    parser.add_argument("--similarity-threshold", type=float,
                        default=DEFAULT_CONFIG["similarity_threshold"])

    parser.add_argument("--verbose", action="store_true",
                        help="Print bursts to console")

    return parser.parse_args()


def main():
    args = parse_args()

    loader = WindowsImageLoader()

    # Strict = ResNet50, Loose = ResNet18
    strict_embedder = DesktopTorchEmbeddingBackend(model_name="resnet50")
    loose_embedder  = DesktopTorchEmbeddingBackend(model_name="resnet18")

    print(f"Scanning folder: {args.input_folder}")
    result = pipeline_bursts_with_similarity(
        loader=loader,
        strict_embedder=strict_embedder,
        loose_embedder=loose_embedder,
        folder_path=args.input_folder,
        time_gap_max=args.time_gap_max,
        min_burst_len=args.min_burst_len,
        similarity_threshold=args.similarity_threshold,
    )

    # כתיבה ל־JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved output to {args.output_json}")

    if args.verbose:
        print("\n=== Bursts Detected ===\n")
        for burst in result["bursts"]:
            print(f"Burst {burst['burst_id']} ({burst['num_images']} images)")
            for p in burst["image_ids"]:
                print(f"   - {p}")
            print(f"   Recommended: {burst['recommended_keep']}\n")


if __name__ == "__main__":
    main()
