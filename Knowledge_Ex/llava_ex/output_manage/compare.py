import safetensors.torch
import torch

def read_safetensors(file_path):
    try:
        # 加载 SafeTensors 文件
        tensors = safetensors.torch.load_file(file_path)
        
        # 打印所有张量的键和形状
        print(f"\nLoaded tensors from {file_path}:")
        for key, tensor in tensors.items():
            print(f"Key: {key}, Shape: {tensor.shape}, Dtype: {tensor.dtype}")
            print(tensor)
            
        return tensors
    except Exception as e:
        print(f"Error loading SafeTensors file {file_path}: {e}")
        return None

def compare_safetensors(tensors1, tensors2, file1_path, file2_path, rtol=1e-5, atol=1e-8):
    if tensors1 is None or tensors2 is None:
        print("Comparison failed: One or both tensor sets are invalid.")
        return False
    
    # 获取键集合
    keys1 = set(tensors1.keys())
    keys2 = set(tensors2.keys())
    
    # 检查键是否一致
    if keys1 != keys2:
        print(f"Key mismatch between {file1_path} and {file2_path}:")
        print(f"Keys in {file1_path} but not in {file2_path}: {keys1 - keys2}")
        print(f"Keys in {file2_path} but not in {file1_path}: {keys2 - keys1}")
        return False
    
    # 比较每个键对应的张量
    all_match = True
    for key in keys1:
        tensor1 = tensors1[key]
        tensor2 = tensors2[key]
        
        # 检查形状
        if tensor1.shape != tensor2.shape:
            print(f"Shape mismatch for key '{key}': {tensor1.shape} vs {tensor2.shape}")
            all_match = False
            continue
        
        # 检查数据类型
        if tensor1.dtype != tensor2.dtype:
            print(f"Dtype mismatch for key '{key}': {tensor1.dtype} vs {tensor2.dtype}")
            all_match = False
            continue
        print(torch.eq(tensor1, tensor2))
        # 检查值（使用 torch.allclose 处理浮点数）
        if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
            print(f"Value mismatch for key '{key}'")
            all_match = False
        else:
            print(f"Tensor '{key}' matches in both files")
    
    if all_match:
        print(f"\nAll tensors match between {file1_path} and {file2_path}")
    else:
        print(f"\nSome tensors do not match between {file1_path} and {file2_path}")
    
    return all_match

if __name__ == "__main__":
    # 替换为你的 SafeTensors 文件路径
    file1_path = "/datanfs2/medllava/llava/Externalization_llava/checkpoints/llava-v1.5-7b-lora-July4-MutiKnowledgeLearning-bs4-extraction-2e-4-epoch11-memorytokens128-knowledge2-Chiwwawa-10000lr_prompt_tuning/llava-prompt_tuning/memorytokens_part1.safetensors"
    file2_path = "/datanfs2/medllava/llava/Externalization_llava/checkpoints/llava-v1.5-7b-lora-July4-MutiKnowledgeLearning-bs4-extraction-2e-4-epoch11-memorytokens128-knowledge2-Chiwwawa-10000lr_prompt_tuning/llava-prompt_tuning/memorytokens_part2.safetensors"
    
    # 读取两个 SafeTensors 文件
    tensors1 = read_safetensors(file1_path)
    tensors2 = read_safetensors(file2_path)
    
    # 比较两个文件
    compare_safetensors(tensors1, tensors2, file1_path, file2_path)
    
    # 示例：访问特定张量
    if tensors1 and "some_tensor_name" in tensors1:
        specific_tensor = tensors1["some_tensor_name"]
        print(f"\nSpecific tensor value from {file1_path}:\n{specific_tensor}")