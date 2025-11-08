import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from itertools import islice

# # 将项目根目录加入路径，保持与 train.py 一致
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.transformer import Transformer  # 与训练脚本保持同路径导入
# from src.utils.data_utils import (
#     load_iwslt2017_data,
#     IWSLT2017Dataset,
# )
# from src.utils.train_utils import LabelSmoothingLoss, evaluate


# def set_seed(seed: int):
#     """设置随机种子以确保结果可复现"""
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     # 确保cudnn的确定性，可能会牺牲一些性能
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# def find_latest_checkpoint(results_dir: str) -> str:
#     """在 `results_dir` 下查找最新的 `exp_*/best_model.pt`"""
#     if not os.path.isdir(results_dir):
#         return ""
#     candidates = []
#     for d in os.listdir(results_dir):
#         p = os.path.join(results_dir, d, "best_model.pt")
#         if d.startswith("exp_") and os.path.isfile(p):
#             try:
#                 num = int(d.split("_")[1])
#                 candidates.append((num, p))
#             except (ValueError, IndexError):
#                 continue
#     if not candidates:
#         return ""
#     candidates.sort(key=lambda x: x[0], reverse=True)
#     return candidates[0][1]


# def greedy_decode(model, src, max_len, sos_idx, eos_idx, device):
#     """
#     使用贪心解码生成译文 (单句)
#     Args:
#         model: 训练好的 Transformer 模型
#         src: 源语言句子张量 [1, src_len]
#         max_len: 生成句子的最大长度
#         sos_idx: <sos> 的索引
#         eos_idx: <eos> 的索引
#         device: 计算设备
#     Returns:
#         Tensor: 生成的目标语言句子ID序列
#     """
#     model.eval()
#     with torch.no_grad():
#         # 1. 创建源的 padding mask
#         # 假设模型有 make_src_mask 方法
#         src_mask = model.make_src_mask(src)
        
#         # 2. 编码器只需要计算一次
#         enc_output, _ = model.encoder(src, src_mask)
        
#         # 3. 初始化解码器输入为 <sos>
#         dec_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
        
#         # 4. 自回归循环
#         for _ in range(max_len - 1):
#             # 创建解码器自注意力掩码
#             # 假设模型有 make_tgt_mask 方法
#             dec_mask = model.make_tgt_mask(dec_input)
            
#             # 解码器前向传播
#             output, _, _ = model.decoder(dec_input, enc_output, dec_mask, src_mask)
            
#             # 5. 取最后一个词的预测结果 (贪心选择)
#             pred_token_id = output.argmax(dim=-1)[:, -1]
            
#             # 6. 拼接到输入序列中
#             dec_input = torch.cat([dec_input, pred_token_id.unsqueeze(0)], dim=1)
            
#             # 7. 如果是 <eos>，则停止
#             if pred_token_id.item() == eos_idx:
#                 break
                
#     return dec_input.squeeze(0) # 移除 batch 维度


# def main():
#     parser = argparse.ArgumentParser(description="Evaluate Transformer on IWSLT2017 test set")

#     # 数据参数
#     parser.add_argument("--dataset", type=str, default="iwslt", choices=["iwslt"], help="Dataset to use")
#     parser.add_argument("--data_dir", type=str, default="../data", help="Data directory for IWSLT dataset")
#     parser.add_argument("--language_pair", type=str, default="de-en", choices=["de-en", "en-de"], help="Language pair")
#     parser.add_argument("--max_len", type=int, default=120, help="Max sequence length")
#     parser.add_argument("--batch_size", type=int, default=256, help="Batch size for loss evaluation (can be larger)")

#     # 模型参数（需与训练时一致）
#     parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
#     parser.add_argument("--n_layers", type=int, default=3, help="Number of layers")
#     parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
#     parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward dimension")
#     parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

#     # 评估/输出参数
#     parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
#     parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint (best_model.pt)")
#     parser.add_argument("--results_dir", type=str, default="../results", help="Root results directory to search ckpt")
#     parser.add_argument("--seed", type=int, default=8, help="Random seed")
#     parser.add_argument("--print_samples", type=int, default=5, help="Number of sample predictions to print")
    
#     parser.add_argument("--beam_width", type=int, default=3, help="Number of sample predictions to print")

#     args = parser.parse_args()

#     # 设备与种子
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     set_seed(args.seed)
#     print(f"Using device: {device}")

#     # 解析checkpoint路径
#     ckpt_path = args.ckpt_path
#     if not ckpt_path:
#         ckpt_path = find_latest_checkpoint(args.results_dir)
#         if ckpt_path:
#             print(f"Auto-selected checkpoint: {ckpt_path}")
#     if not ckpt_path or not os.path.isfile(ckpt_path):
#         raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'. Please set --ckpt_path.")

#     # 加载数据与词表
#     print(f"Loading IWSLT2017 ({args.language_pair}) for evaluation...")
#     _, _, _, _, test_src, test_tgt, src_vocab, tgt_vocab = load_iwslt2017_data(
#         data_dir=args.data_dir, language_pair=args.language_pair
#     )
#     test_dataset = IWSLT2017Dataset(test_src, test_tgt, src_vocab, tgt_vocab, max_len=args.max_len)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
#     src_vocab_size = len(src_vocab)
#     tgt_vocab_size = len(tgt_vocab)

#     # 构建模型并加载权重
#     print("Building Transformer model...")
#     model = Transformer(
#         src_vocab_size=src_vocab_size,
#         tgt_vocab_size=tgt_vocab_size,
#         d_model=args.d_model,
#         n_layers=args.n_layers,
#         n_heads=args.n_heads,
#         d_ff=args.d_ff,
#         dropout=args.dropout,
#         pad_idx=src_vocab.get("<pad>", 0) # 假设 pad_idx 相同
#     ).to(device)

#     print(f"Loading checkpoint: {ckpt_path}")
#     state = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(state["model_state_dict"])

#     # 评估损失和困惑度
#     criterion = LabelSmoothingLoss(smoothing=args.label_smoothing, ignore_index=src_vocab.get("<pad>", 0))
#     print("Evaluating loss on test set...")
#     test_loss = evaluate(model, test_loader, criterion, device)
#     perplexity = float(np.exp(test_loss))

#     print(f"Test Loss: {test_loss:.4f}")
#     print(f"Perplexity: {perplexity:.4f}")

#     # 保存测试指标
#     out_dir = os.path.dirname(ckpt_path)
#     metrics_path = os.path.join(out_dir, "test_metrics.json")
#     with open(metrics_path, "w", encoding="utf-8") as f:
#         json.dump({"test_loss": test_loss, "perplexity": perplexity}, f, ensure_ascii=False, indent=2)
#     print(f"Saved metrics to {metrics_path}")

#     # 打印样本翻译
#     # if args.print_samples > 0:
#     #     print("\n--- Translating Samples ---")
        
#     #     pad_idx = src_vocab.get("<pad>", 0)
#     #     sos_idx = tgt_vocab.get("<sos>", -1)
#     #     eos_idx = tgt_vocab.get("<eos>", -1)

#     #     if sos_idx == -1 or eos_idx == -1:
#     #         print("Error: <sos> or <eos> not in target vocabulary. Cannot generate samples.")
#     #         return

#     #     inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
#     #     inv_src_vocab = {v: k for k, v in src_vocab.items()}

#     #     # 从测试数据集中随机选择样本
#     #     sample_indices = np.random.choice(len(test_dataset), args.print_samples, replace=False)
        
#     #     for i, idx in enumerate(sample_indices):
#     #         sample = test_dataset[idx]
#     #         src_tensor = sample["src"].unsqueeze(0).to(device) # [1, src_len]
            
#     #         # 调用贪心解码函数
#     #         pred_ids = greedy_decode(model, src_tensor, args.max_len, sos_idx, eos_idx, device)
            
#     #         # --- 格式化并打印结果 ---
#     #         src_tokens = [inv_src_vocab.get(t.item(), "<unk>") for t in sample["src"] if t.item() != pad_idx]
#     #         ref_tokens = [inv_tgt_vocab.get(t.item(), "<unk>") for t in sample["tgt"] if t.item() not in [pad_idx, sos_idx, eos_idx]]
#     #         pred_tokens = [inv_tgt_vocab.get(t.item(), "<unk>") for t in pred_ids if t.item() not in [sos_idx, eos_idx]]
            
#     #         print(f"\n--- Sample {i+1}/{args.print_samples} ---")
#     #         print("SRC: ", " ".join(src_tokens))
#     #         print("REF: ", " ".join(ref_tokens))
#     #         print("PRED:", " ".join(pred_tokens))
#     if args.print_samples > 0:
#         print("\n--- Translating Samples ---")
        
#         pad_idx = src_vocab.get("<pad>", 0)
#         sos_idx = tgt_vocab.get("<sos>", -1)
#         eos_idx = tgt_vocab.get("<eos>", -1)

#         inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
#         inv_src_vocab = {v: k for k, v in src_vocab.items()}

#         # 从测试数据集中随机选择样本
#         sample_indices = np.random.choice(len(test_dataset), args.print_samples, replace=False)
        
#         model.eval() # 确保模型在评估模式
#         with torch.no_grad():
#             for i, idx in enumerate(sample_indices):
#                 sample = test_dataset[idx]
#                 src_tensor = sample["src"].unsqueeze(0).to(device) # [1, src_len]
                
#                 # --- ✨ 直接调用模型内置的 generate 方法 ---
#                 pred_ids = model.generate(src_tensor, max_len=args.max_len)
#                 # pred_ids = model.beam_search_decode(src_tensor, beam_width=args.beam_width, max_len=args.max_len)
#                 pred_ids = pred_ids.squeeze(0) # 移除 batch 维度
                
#                 # --- 格式化并打印结果 ---
#                 src_tokens = [inv_src_vocab.get(t.item(), "<unk>") for t in sample["src"] if t.item() != pad_idx]
#                 ref_tokens = [inv_tgt_vocab.get(t.item(), "<unk>") for t in sample["tgt"] if t.item() not in [pad_idx, sos_idx, eos_idx]]
                
#                 # 从预测结果中移除 <sos> 和 <eos>
#                 pred_tokens = []
#                 for t_id in pred_ids.tolist():
#                     if t_id == sos_idx:
#                         continue
#                     if t_id == eos_idx:
#                         break
#                     pred_tokens.append(inv_tgt_vocab.get(t_id, "<unk>"))

#                 print(f"\n--- Sample {i+1}/{args.print_samples} ---")
#                 print("SRC: ", " ".join(src_tokens))
#                 print("REF: ", " ".join(ref_tokens))
#                 print("PRED:", " ".join(pred_tokens))

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer
from src.utils.data_utils import load_iwslt2017_data, IWSLT2017Dataset
from src.utils.train_utils import LabelSmoothingLoss, evaluate

# set_seed 和 find_latest_checkpoint 函数保持不变
def set_seed(seed: int):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_latest_checkpoint(results_dir: str) -> str:
    """在 `results_dir` 下查找最新的 `exp_*/best_model.pt`"""
    if not os.path.isdir(results_dir): return ""
    candidates = []
    for d in os.listdir(results_dir):
        p = os.path.join(results_dir, d, "best_model.pt")
        if d.startswith("exp_") and os.path.isfile(p):
            try:
                num = int(d.split("_")[1])
                candidates.append((num, p))
            except (ValueError, IndexError): continue
    if not candidates: return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

# --- 主要修改在这里 ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer on IWSLT2017 test set")

    # ... (所有数据和模型参数保持不变) ...
    parser.add_argument("--dataset", type=str, default="iwslt", choices=["iwslt"])
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--language_pair", type=str, default="de-en")
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # --- ✨ 新增和修改的评估参数 ✨ ---
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="../results")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--print_samples", type=int, default=5)
    
    # 新增解码策略选项
    parser.add_argument("--decode_strategy", type=str, default="beam", 
                        choices=["greedy", "beam"], help="Decoding strategy")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search")
    parser.add_argument("--length_penalty", type=float, default=0.6, help="Length penalty for beam search")

    args = parser.parse_args()

    # 设备与种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")

    # ... (加载 checkpoint 和数据的逻辑保持不变) ...
    ckpt_path = args.ckpt_path
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint(args.results_dir)
        if ckpt_path: print(f"Auto-selected checkpoint: {ckpt_path}")
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'. Please set --ckpt_path.")

    print(f"Loading IWSLT2017 ({args.language_pair}) for evaluation...")
    _, _, _, _, test_src, test_tgt, src_vocab, tgt_vocab = load_iwslt2017_data(
        data_dir=args.data_dir, language_pair=args.language_pair
    )
    test_dataset = IWSLT2017Dataset(test_src, test_tgt, src_vocab, tgt_vocab, max_len=args.max_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # 获取特殊 token 的索引
    pad_idx = src_vocab.get("<pad>", 0)
    sos_idx = tgt_vocab.get("<sos>", 2) # 假设和训练时一致
    eos_idx = tgt_vocab.get("<eos>", 3) # 假设和训练时一致

    # 构建模型并加载权重
    print("Building Transformer model...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_idx=pad_idx,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
    ).to(device)

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    # ... (评估损失和困惑度的逻辑保持不变) ...
    criterion = LabelSmoothingLoss(smoothing=args.label_smoothing, ignore_index=pad_idx)
    print("Evaluating loss on test set...")
    test_loss = evaluate(model, test_loader, criterion, device)
    perplexity = float(np.exp(test_loss))
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    
    out_dir = os.path.dirname(ckpt_path)
    metrics_path = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"test_loss": test_loss, "perplexity": perplexity}, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # --- ✨ 全新、更灵活的样本翻译部分 ✨ ---
    if args.print_samples > 0:
        print(f"\n--- Translating Samples (Strategy: {args.decode_strategy}) ---")
        
        inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
        inv_src_vocab = {v: k for k, v in src_vocab.items()}

        sample_indices = np.random.choice(len(test_dataset), args.print_samples, replace=False)
        
        model.eval()
        with torch.no_grad():
            for i, idx in enumerate(sample_indices):
                sample = test_dataset[idx]
                src_tensor = sample["src"].unsqueeze(0).to(device) # [1, src_len]
                
                # print(src_tensor)
                # --- 根据策略选择解码方法 ---
                if args.decode_strategy == "beam":
                    pred_ids = model.beam_search_decode(
                        src_tensor, 
                        beam_width=args.beam_width, 
                        max_len=args.max_len,
                        length_penalty=args.length_penalty
                    )
                    print(pred_ids)
                else: # greedy
                    # 调用模型内置的贪心解码方法
                    pred_ids_with_sos = model.generate(src_tensor, max_len=args.max_len)
                    pred_ids = pred_ids_with_sos.squeeze(0) # 移除 batch 维度
                
                # --- 格式化并打印结果 ---
                src_tokens = [inv_src_vocab.get(t.item(), "<unk>") for t in sample["src"] if t.item() != pad_idx]
                ref_tokens = [inv_tgt_vocab.get(t.item(), "<unk>") for t in sample["tgt"] if t.item() not in [pad_idx, sos_idx, eos_idx]]
                
                # 从预测结果中移除 <sos> 和 <eos>
                pred_tokens = []
                for t_id in pred_ids.tolist():
                    # 跳过句首的 <sos>
                    if t_id == sos_idx:
                        continue
                    # 遇到 <eos> 就结束
                    if t_id == eos_idx:
                        break
                    pred_tokens.append(inv_tgt_vocab.get(t_id, "<unk>"))

                print(f"\n--- Sample {i+1}/{args.print_samples} ---")
                print("SRC: ", " ".join(src_tokens))
                print("REF: ", " ".join(ref_tokens))
                print("PRED:", " ".join(pred_tokens))

if __name__ == "__main__":
    main()