import json
import sys
import os
import shutil
import argparse
# 用法: python update_adapter.py <divisor>
# if len(sys.argv) != 2:
#     print("用法: python update_adapter.py <divisor>")
#     sys.exit(1)


parser = argparse.ArgumentParser(description="Convert .bin to .safetensors with key renaming.")
parser.add_argument('--knowledge_num', type=str, required=True, help='Path to input .bin file')
parser.add_argument('--model_dir', type=str, required=True, help='model_dir')
parser.add_argument('--eval_stage', type=str, default='1', help='retain_full')
args = parser.parse_args()




knowledge_num = int(args.knowledge_num)
model_dir = args.model_dir
eval_stage = int(args.eval_stage)
json_path = os.path.join(model_dir,"adapter_config.json")
# 1. 修改json
if eval_stage == 1:
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "num_virtual_tokens" not in config:
        print("未找到num_virtual_tokens字段")
        sys.exit(1)

    old_value = config["num_virtual_tokens"]
    # new_value = old_value // knowledge_num
    new_value = old_value
    config["num_virtual_tokens"] = new_value

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f'"num_virtual_tokens" 已从 {old_value} 改为 {new_value}')

    # 2. 重命名adapter_model.safetensors为adapter_model_full.safetensors
    src_model = os.path.join(model_dir, "adapter_model.safetensors")
    dst_model_full = os.path.join(model_dir, "adapter_model_full.safetensors")
    if os.path.exists(src_model):
        os.rename(src_model, dst_model_full)
        print(f"{src_model} 已重命名为 {dst_model_full}")
    else:
        print(f"{src_model} 不存在，跳过重命名")

# 3. 复制memorytokens_part1.safetensors为adapter_model.safetensors
src_memory = os.path.join(model_dir, f"memorytokens_part{eval_stage}.safetensors")
dst_model = os.path.join(model_dir, "adapter_model.safetensors")
if os.path.exists(src_memory):
    shutil.copy(src_memory, dst_model)
    print(f"{src_memory} 已复制为 {dst_model}")
else:
    print(f"{src_memory} 不存在，无法复制")