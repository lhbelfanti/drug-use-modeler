import os
import glob
import argparse
import onnxruntime as ort


def inspect(model_path, base_dir):
    rel = os.path.relpath(model_path, base_dir)
    session = ort.InferenceSession(model_path)
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"=== {rel} ===")
    print("  Inputs:")
    for inp in inputs:
        print(f"    [{inp.name}]  shape={inp.shape}  type={inp.type}")
    print("  Outputs:")
    for out in outputs:
        print(f"    [{out.name}]  shape={out.shape}  type={out.type}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect ONNX model inputs and outputs.")
    parser.add_argument(
        "--models-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
        help="Root directory containing .onnx files (default: ../models relative to this script)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only inspect paths containing this substring (e.g. 'bert_base', 'raw-corpus/ffn')",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.models_dir)
    paths = sorted(glob.glob(os.path.join(base_dir, "**", "*.onnx"), recursive=True))

    if args.filter:
        paths = [p for p in paths if args.filter in p]

    if not paths:
        print(f"No .onnx files found under {base_dir}")
        return

    for path in paths:
        inspect(path, base_dir)


if __name__ == "__main__":
    main()
