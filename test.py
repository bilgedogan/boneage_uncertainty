from pandas.core import construction
import argparse
import torch

from data_loader import load_data, build_val_or_test_loader
from model import build_multi_input_model
from metrics import evaluate_and_save_metrics
from config import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained bone age model on the held-out TEST set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth model weights")
    parser.add_argument("--backbone", type=str, default="efficientnet_b3", help="backbone type")
    parser.add_argument("--seed", type=int, default=0, help="Torch init/training seed (data split stays fixed)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_name = f"seed{args.seed}" + ("_quicktest" if args.quick_test else "")
    run_dir = OUTPUT_DIR / args.backbone / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    # Re-uses the same deterministic split (RANDOM_STATE) as training,
    # so max_age matches the normalization used at train time.
    _, _, _, test_df, max_age = load_data(sample_frac=1.0, seed=args.seed)
    print(f"Test set: {len(test_df)} samples")

    test_loader = build_val_or_test_loader(test_df, backbone_name=args.backbone)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model weights from {args.checkpoint}")
    model = build_multi_input_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    evaluate_and_save_metrics(
        model, test_loader, test_df, max_age,
        run_dir, args.seed, device=device, split="test",
    )
    print("Test evaluation completed.")

if __name__ == "__main__":
    main()
