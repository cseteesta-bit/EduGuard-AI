import argparse
import json
from eduguard.data import load_oulad, synthetic_data
from eduguard.model import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EduGuard-AI")
    parser.add_argument("--data-dir", default="data/oulad")
    parser.add_argument("--demo", action="store_true", help="Train synthetic demonstration data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    df = synthetic_data(seed=args.seed) if args.demo else load_oulad(args.data_dir)
    print(json.dumps(train(df, args.model_dir, seed=args.seed, alpha=args.alpha, demo=args.demo), indent=2))
