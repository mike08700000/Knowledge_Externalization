import torch
from safetensors.torch import save_file, load_file
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert .bin to .safetensors with key renaming.")
    parser.add_argument('--bin_file', type=str, required=True, help='Path to input .bin file')
    parser.add_argument('--safetensors_file', type=str, required=True, help='Path to output .safetensors file')
    parser.add_argument('--compare_file', type=str, default=None, help='Path to reference .safetensors file (optional)')
    args = parser.parse_args()

    state_dict = torch.load(args.bin_file, map_location="cpu")
    i = 0
    j = 0
    new_state_dict = {}
    for key in list(state_dict.keys()):
        j += 1
        if "module." in key:
            i += 1
            new_key = key.replace("module.", "")
            new_key = new_key.replace("base_model.base_model.", "base_model.")
            new_key = new_key.replace(".default.", ".")
            new_state_dict[new_key] = state_dict[key]
            # print(f"Renaming key: {key} to {new_key}")
    print(f"Total keys: {j}, Filtered keys: {i}")
    if args.compare_file:
        state_dict_good = load_file(args.compare_file)
        print()
        p = 0
        for key in list(state_dict_good.keys()):
            p += 1
    save_file(new_state_dict, args.safetensors_file)
    print(f"Successfully converted {args.bin_file} to {args.safetensors_file}")

if __name__ == "__main__":
    main()