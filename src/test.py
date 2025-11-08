import argparse
import os
import sys
import json
import torch
import numpy as np

# 将项目根目录加入路径，保持与 train.py 一致
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer  # 与训练脚本保持同路径导入
from src.utils.data_utils import (
    load_iwslt2017_data,
    IWSLT2017Dataset,
)
from src.utils.train_utils import LabelSmoothingLoss, evaluate


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def find_latest_checkpoint(results_dir: str) -> str:
    """
    在 `results_dir` 下查找最新的 `exp_*/best_model.pt`，找不到则返回空字符串。
    """
    if not os.path.isdir(results_dir):
        return ""
    candidates = []
    for d in os.listdir(results_dir):
        p = os.path.join(results_dir, d, "best_model.pt")
        if d.startswith("exp_") and os.path.isfile(p):
            # 以编号排序（优先选择编号最大的）
            try:
                num = int(d.split("_")[1])
            except Exception:
                num = -1
            candidates.append((num, p))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def ids_to_tokens(ids: torch.Tensor, vocab: dict) -> list:
    """
    将一维id序列转换为token列表。若id不在词表，则返回'<unk>'。
    """
    # 构造反向词表
    inv = {v: k for k, v in vocab.items()}
    tokens = []
    for i in ids.tolist():
        tokens.append(inv.get(i, "<unk>"))
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer on toy or IWSLT2017 test set")

    # 数据参数
    parser.add_argument("--dataset", type=str, default="iwslt", choices=["toy", "iwslt"], help="Dataset to use")
    parser.add_argument("--data_dir", type=str, default="../data", help="Data directory for IWSLT dataset")
    parser.add_argument("--language_pair", type=str, default="de-en", choices=["de-en", "en-de"], help="Language pair")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to use (debug)")
    parser.add_argument("--n_samples", type=int, default=1000, help="Toy dataset samples")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Toy vocab size")
    parser.add_argument("--max_len", type=int, default=120, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")

    # 模型参数（需与训练时一致）
    parser.add_argument("--d_model", type=int, default=32, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=128, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # 评估/输出参数
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint (best_model.pt)")
    parser.add_argument("--results_dir", type=str, default="../results", help="Root results directory to search ckpt")
    parser.add_argument("--seed", type=int, default=8, help="Random seed")
    parser.add_argument("--print_samples", type=int, default=5, help="Number of sample predictions to print (iwslt only)")

    args = parser.parse_args()

    # 设备与种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Using device: {device}")

    # 解析checkpoint路径
    ckpt_path = args.ckpt_path
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint(args.results_dir)
        if ckpt_path:
            print(f"Auto-selected checkpoint: {ckpt_path}")
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError("Checkpoint not found. Please set --ckpt_path to a valid 'best_model.pt'.")

    # 构造数据与词表
    print(f"Loading IWSLT2017 ({args.language_pair}) for evaluation...")
    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt, src_vocab, tgt_vocab = load_iwslt2017_data(
        data_dir=args.data_dir,
        language_pair=args.language_pair,
        max_samples=args.max_samples,
    )
    # 测试集加载器
    test_dataset = IWSLT2017Dataset(test_src, test_tgt, src_vocab, tgt_vocab, max_len=args.max_len)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

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
    ).to(device)

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])  # 与训练保存格式一致

    # 损失函数与评估
    criterion = LabelSmoothingLoss(smoothing=args.label_smoothing)
    print("Evaluating on test set...")
    test_loss = evaluate(model, test_loader, criterion, device)
    perplexity = float(np.exp(test_loss))

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

    # 保存测试指标到checkpoint所在目录
    out_dir = os.path.dirname(ckpt_path)
    metrics_path = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"test_loss": test_loss, "perplexity": perplexity}, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")

    if args.dataset == "iwslt" and args.print_samples > 0:
        model.eval()
        from itertools import islice
        inv_tgt = {v: k for k, v in tgt_vocab.items()}
        inv_src = {v: k for k, v in src_vocab.items()}
        with torch.no_grad():
            printed = 0
            for batch in test_loader:
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                output, *_ = model(src, tgt)
                pred_ids = output.argmax(dim=-1)
                for i in range(src.size(0)):
                    if printed >= args.print_samples:
                        break
                    src_tokens = [inv_src.get(x.item(), "<unk>") for x in batch["src"][i] if x.item() != src_vocab.get("<pad>", 0)]
                    tgt_ref = [inv_tgt.get(x.item(), "<unk>") for x in batch["tgt"][i] if x.item() != tgt_vocab.get("<pad>", 0)]
                    tgt_pred = [inv_tgt.get(x.item(), "<unk>") for x in pred_ids[i] if x.item() != tgt_vocab.get("<pad>", 0)]
                    print("--- Sample ---")
                    print("SRC:", " ".join(src_tokens))
                    print("REF:", " ".join(tgt_ref))
                    print("PRED:", " ".join(tgt_pred))
                    printed += 1
                if printed >= args.print_samples:
                    break


if __name__ == "__main__":
    main()