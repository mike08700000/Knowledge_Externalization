from safetensors.torch import load_file

# 指定safetensors文件路径
file_path = "/datanfs2/medllava/llava/Externalization_llava/checkpoints/llava-v1.5-7b-lora-July4-MutiKnowledgeLearning-bs4-extraction-1e-4-epoch1-memorytokens64-knowledge2/llava-prompt_tuning/adapter_model_full.safetensors"

# 读取safetensors文件
state_dict = load_file(file_path)

# 打印所有参数名
for key in state_dict:
    print(key, state_dict[key].shape)