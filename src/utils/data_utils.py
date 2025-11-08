# import torch
# import numpy as np
# import os
# import re
# import xml.etree.ElementTree as ET
# from torch.utils.data import Dataset, DataLoader
# from collections import Counter

# class IWSLT2017Dataset(Dataset):
#     """
#     IWSLT2017数据集类，用于处理真实翻译数据
#     """
#     def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab, max_len=100):
#         """
#         初始化数据集
        
#         Args:
#             src_data: 源语言文本数据列表
#             tgt_data: 目标语言文本数据列表
#             src_vocab: 源语言词汇表
#             tgt_vocab: 目标语言词汇表
#             max_len: 最大序列长度
#         """
#         self.src_data = src_data
#         self.tgt_data = tgt_data

#         self.src_vocab = src_vocab
#         self.tgt_vocab = tgt_vocab
#         self.max_len = max_len
        
#     def __len__(self):
#         return len(self.src_data)
    
#     def __getitem__(self, idx):
#         src_tokens = self.tokenize_and_numericalize(self.src_data[idx], self.src_vocab, self.max_len)
#         tgt_tokens = self.tokenize_and_numericalize(self.tgt_data[idx], self.tgt_vocab, self.max_len, add_sos=True)
        
#         return {
#             'src': src_tokens,
#             'tgt': tgt_tokens
#         }
    
#     def tokenize_and_numericalize(self, text, vocab, max_len, add_sos=False):
#         """
#         将文本分词并转换为数字序列
        
#         Args:
#             text: 输入文本
#             vocab: 词汇表
#             max_len: 最大序列长度
#             add_sos: 是否添加开始符号
            
#         Returns:
#             tokens: 数字序列
#         """
#         # 简单分词，按空格分割
#         words = text.strip().lower().split()
        
#         # 转换为数字序列
#         if add_sos:
#             tokens = [vocab['<sos>']] + [vocab.get(w, vocab['<unk>']) for w in words] + [vocab['<eos>']]
#         else:
#             tokens = [vocab.get(w, vocab['<unk>']) for w in words] + [vocab['<eos>']]
        
#         # 截断或填充到最大长度
#         if len(tokens) > max_len:
#             tokens = tokens[:max_len]
#         else:
#             tokens = tokens + [vocab['<pad>']] * (max_len - len(tokens))
        
#         return torch.tensor(tokens, dtype=torch.long)
    
# def load_iwslt2017_data(data_dir, language_pair='de-en', max_samples=None):
#     """
#     加载IWSLT2017数据集
    
#     Args:
#         data_dir: 数据集目录
#         language_pair: 语言对，如'de-en'表示德语到英语
#         max_samples: 最大样本数，用于调试
        
#     Returns:
#         train_src: 训练集源语言文本列表
#         train_tgt: 训练集目标语言文本列表
#         val_src: 验证集源语言文本列表
#         val_tgt: 验证集目标语言文本列表
#         test_src: 测试集源语言文本列表
#         test_tgt: 测试集目标语言文本列表
#         src_vocab: 源语言词汇表
#         tgt_vocab: 目标语言词汇表
#     """
#     src_lang, tgt_lang = language_pair.split('-')
#     data_path = os.path.join(data_dir, language_pair)
    
#     # 加载训练数据
#     train_src_path = os.path.join(data_path, f'train.tags.{language_pair}.{src_lang}')
#     train_tgt_path = os.path.join(data_path, f'train.tags.{language_pair}.{tgt_lang}')
    
#     # 加载验证数据
#     val_src_path = os.path.join(data_path, f'IWSLT17.TED.dev2010.{language_pair}.{src_lang}.xml')
#     val_tgt_path = os.path.join(data_path, f'IWSLT17.TED.dev2010.{language_pair}.{tgt_lang}.xml')
    
#     # 加载测试数据
#     test_src_path = os.path.join(data_path, f'IWSLT17.TED.tst2010.{language_pair}.{src_lang}.xml')
#     test_tgt_path = os.path.join(data_path, f'IWSLT17.TED.tst2010.{language_pair}.{tgt_lang}.xml')
    
#     # 解析训练数据
#     train_src = []
#     train_tgt = []
    
#     with open(train_src_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             # 跳过XML标签行
#             if line.startswith(' '):
#                 line = line[1:]  # 去除开头的空格
                
#             if not line.startswith('<'):
#                 train_src.append(line.strip())
    
#     with open(train_tgt_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             # 跳过XML标签行
#             if not line.startswith('<'):
#                 train_tgt.append(line.strip())
    
#     # 确保源语言和目标语言数据长度一致
#     assert len(train_src) == len(train_tgt), "训练数据源语言和目标语言长度不一致"
    
#     # 解析验证和测试数据
#     val_src = parse_xml_file(val_src_path)
#     val_tgt = parse_xml_file(val_tgt_path)
#     test_src = parse_xml_file(test_src_path)
#     test_tgt = parse_xml_file(test_tgt_path)
    
#     print("max_samples:", max_samples)
    
#     # 限制样本数量（用于调试）
#     if max_samples:
#         train_src = train_src[:max_samples]
#         train_tgt = train_tgt[:max_samples]
#         val_src = val_src[:max_samples//10]
#         val_tgt = val_tgt[:max_samples//10]
#         test_src = test_src[:max_samples//10]
#         test_tgt = test_tgt[:max_samples//10]
    
#     # 构建词汇表
#     src_vocab = build_vocab(train_src)
#     tgt_vocab = build_vocab(train_tgt)
    
#     return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt, src_vocab, tgt_vocab

# def parse_xml_file(file_path):
#     """
#     解析XML格式的数据文件
    
#     Args:
#         file_path: XML文件路径
        
#     Returns:
#         sentences: 句子列表
#     """
#     sentences = []
#     try:
#         tree = ET.parse(file_path)
#         root = tree.getroot()
        
#         # 查找所有文本内容
#         for doc in root.findall('.//doc'):
#             for seg in doc.findall('.//seg'):
#                 if seg.text:
#                     sentences.append(seg.text.strip())
#     except Exception as e:
#         print(f"解析XML文件出错: {e}")
    
#     return sentences

# def build_vocab(texts, min_freq=2, max_size=50000):
#     """
#     构建词汇表
    
#     Args:
#         texts: 文本列表
#         min_freq: 最小词频
#         max_size: 最大词汇表大小
        
#     Returns:
#         vocab: 词汇表字典，将单词映射到索引
#     """
#     # 统计词频
#     counter = Counter()
#     for text in texts:
#         words = text.strip().lower().split()
#         counter.update(words)
    
#     # 过滤低频词并限制词汇表大小
#     word_counts = [(w, c) for w, c in counter.items() if c >= min_freq]
#     word_counts.sort(key=lambda x: x[1], reverse=True)
#     if max_size:
#         word_counts = word_counts[:max_size]
    
#     # 构建词汇表
#     vocab = {}
#     vocab['<pad>'] = 0  # 填充符号
#     vocab['<unk>'] = 1  # 未知词符号
#     vocab['<sos>'] = 2  # 句子开始符号
#     vocab['<eos>'] = 3  # 句子结束符号
    
#     for i, (word, _) in enumerate(word_counts):
#         vocab[word] = i + 4
    
#     return vocab

# def create_iwslt_dataloaders(train_src, train_tgt, val_src, val_tgt, src_vocab, tgt_vocab, 
#                             batch_size=32, max_len=100):
#     """
#     为IWSLT2017数据集创建数据加载器
    
#     Args:
#         train_src: 训练集源语言文本列表
#         train_tgt: 训练集目标语言文本列表
#         val_src: 验证集源语言文本列表
#         val_tgt: 验证集目标语言文本列表
#         src_vocab: 源语言词汇表
#         tgt_vocab: 目标语言词汇表
#         batch_size: 批次大小
#         max_len: 最大序列长度
        
#     Returns:
#         train_loader: 训练数据加载器
#         val_loader: 验证数据加载器
#     """
#     # 创建训练集和验证集
#     train_dataset = IWSLT2017Dataset(train_src, train_tgt, src_vocab, tgt_vocab, max_len)
#     val_dataset = IWSLT2017Dataset(val_src, val_tgt, src_vocab, tgt_vocab, max_len)
    
#     # 创建数据加载器
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
#     return train_loader, val_loader
    
#     # 创建数据集
#     train_dataset = SimpleDataset(src_data[train_indices], tgt_data[train_indices])
#     val_dataset = SimpleDataset(src_data[val_indices], tgt_data[val_indices])
    
#     # 创建数据加载器
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
#     return train_loader, val_loader
#     val_dataset = SimpleDataset(src_data[val_indices], tgt_data[val_indices])
    
#     # 创建数据加载器
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
#     return train_loader, val_loader

# def translate_sentence(model, src_tensor, src_vocab, tgt_vocab, max_len, device):
#     """
#     使用贪婪解码翻译句子

#     Args:
#         model: 训练好的Transformer模型
#         src_tensor: 源语言句子的张量 [1, src_len]
#         src_vocab: 源语言词汇表
#         tgt_vocab: 目标语言词汇表
#         max_len: 最大生成长度
#         device: 设备

#     Returns:
#         translated_sentence: 翻译后的句子
#     """
#     model.eval()
#     src_mask = model.make_src_mask(src_tensor)
    
#     with torch.no_grad():
#         enc_src = model.encoder(src_tensor, src_mask)

#     tgt_indexes = [tgt_vocab['<sos>']]

#     for i in range(max_len):
#         tgt_tensor = torch.LongTensor(tgt_indexes).unsqueeze(0).to(device)
#         tgt_mask = model.make_tgt_mask(tgt_tensor)
        
#         with torch.no_grad():
#             output, attention = model.decoder(tgt_tensor, enc_src, tgt_mask, src_mask)
        
#         pred_token = output.argmax(2)[:, -1].item()
#         tgt_indexes.append(pred_token)

#         if pred_token == tgt_vocab['<eos>']:
#             break

#     tgt_tokens = [list(tgt_vocab.keys())[list(tgt_vocab.values()).index(i)] for i in tgt_indexes]
#     return " ".join(tgt_tokens[1:-1])


import torch
import numpy as np
import os
import re
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.utils.data.distributed import DistributedSampler
import random

class SimpleDataset(Dataset):
    """
    简单的数据集类，用于小数据集验证
    """
    def __init__(self, src_data, tgt_data):
        """
        初始化数据集
        
        Args:
            src_data: 源语言数据 [n_samples, seq_len]
            tgt_data: 目标语言数据 [n_samples, seq_len]
        """
        self.src_data = src_data
        self.tgt_data = tgt_data
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return {
            'src': self.src_data[idx],
            'tgt': self.tgt_data[idx]
        }
        
class IWSLT2017Dataset(Dataset):
    """
    IWSLT2017数据集类，用于处理真实翻译数据
    """
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab, max_len=100):
        """
        初始化数据集
        
        Args:
            src_data: 源语言文本数据列表
            tgt_data: 目标语言文本数据列表
            src_vocab: 源语言词汇表
            tgt_vocab: 目标语言词汇表
            max_len: 最大序列长度
        """
        # 确保数据集长度一致，取两者的最小长度
        min_len = min(len(src_data), len(tgt_data))
        self.src_data = src_data[:min_len]
        self.tgt_data = tgt_data[:min_len]
        print(f"数据集长度调整为: {min_len} (源语言原长度: {len(src_data)}, 目标语言原长度: {len(tgt_data)})")
        
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        src_tokens = self.tokenize_and_numericalize(self.src_data[idx], self.src_vocab, self.max_len)
        tgt_tokens = self.tokenize_and_numericalize(self.tgt_data[idx], self.tgt_vocab, self.max_len, add_sos=True)
        
        return {
            'src': src_tokens,
            'tgt': tgt_tokens
        }
    
    def tokenize_and_numericalize(self, text, vocab, max_len, add_sos=False):
        """
        将文本分词并转换为数字序列
        
        Args:
            text: 输入文本
            vocab: 词汇表
            max_len: 最大序列长度
            add_sos: 是否添加开始符号
            
        Returns:
            tokens: 数字序列
        """
        # 简单分词，按空格分割
        words = text.strip().lower().split()
        
        # 转换为数字序列
        if add_sos:
            tokens = [vocab['<sos>']] + [vocab.get(w, vocab['<unk>']) for w in words] + [vocab['<eos>']]
        else:
            tokens = [vocab.get(w, vocab['<unk>']) for w in words] + [vocab['<eos>']]
        
        # 截断或填充到最大长度
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [vocab['<pad>']] * (max_len - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)

def create_toy_data(n_samples=1000, src_vocab_size=1000, tgt_vocab_size=1000, 
                   max_len=20, pad_idx=0, sos_idx=2, eos_idx=3):
    """
    创建玩具数据集用于测试
    
    Args:
        n_samples: 样本数量
        src_vocab_size: 源语言词汇表大小
        tgt_vocab_size: 目标语言词汇表大小
        max_len: 最大序列长度
        pad_idx: padding的索引值
        sos_idx: 开始符号的索引值
        eos_idx: 结束符号的索引值
        
    Returns:
        src_data: 源语言数据 [n_samples, seq_len]
        tgt_data: 目标语言数据 [n_samples, seq_len]
    """
    # 创建随机源语言数据
    src_lens = np.random.randint(5, max_len - 2, size=n_samples)
    src_data = []
    for i in range(n_samples):
        # 随机生成序列，避开特殊标记
        seq = np.random.randint(4, src_vocab_size, size=src_lens[i])
        # 添加EOS标记
        seq = np.append(seq, eos_idx)
        # 填充到最大长度
        padded_seq = np.full(max_len, pad_idx)
        padded_seq[:len(seq)] = seq
        src_data.append(padded_seq)
    
    # 创建随机目标语言数据
    tgt_lens = np.random.randint(5, max_len - 2, size=n_samples)
    tgt_data = []
    for i in range(n_samples):
        # 随机生成序列，避开特殊标记
        seq = np.random.randint(4, tgt_vocab_size, size=tgt_lens[i])
        # 添加SOS和EOS标记
        seq = np.concatenate(([sos_idx], seq, [eos_idx]))
        # 填充到最大长度
        padded_seq = np.full(max_len, pad_idx)
        padded_seq[:len(seq)] = seq
        tgt_data.append(padded_seq)
    
    return torch.tensor(src_data, dtype=torch.long), torch.tensor(tgt_data, dtype=torch.long)

def create_dataloaders(src_data, tgt_data, batch_size=32, train_ratio=0.8, shuffle=True):
    """
    创建训练和验证数据加载器
    
    Args:
        src_data: 源语言数据 [n_samples, seq_len]
        tgt_data: 目标语言数据 [n_samples, seq_len]
        batch_size: 批次大小
        train_ratio: 训练集比例
        shuffle: 是否打乱数据
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    n_samples = len(src_data)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    # 划分训练集和验证集
    train_size = int(n_samples * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 创建数据集
    train_dataset = SimpleDataset(src_data[train_indices], tgt_data[train_indices])
    val_dataset = SimpleDataset(src_data[val_indices], tgt_data[val_indices])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
    
def load_iwslt2017_data(data_dir, language_pair='de-en', max_samples=None):
    """
    加载IWSLT2017数据集
    
    Args:
        data_dir: 数据集目录
        language_pair: 语言对，如'de-en'表示德语到英语
        max_samples: 最大样本数，用于调试
        
    Returns:
        train_src: 训练集源语言文本列表
        train_tgt: 训练集目标语言文本列表
        val_src: 验证集源语言文本列表
        val_tgt: 验证集目标语言文本列表
        test_src: 测试集源语言文本列表
        test_tgt: 测试集目标语言文本列表
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
    """
    src_lang, tgt_lang = language_pair.split('-')
    data_path = os.path.join(data_dir, language_pair)
    
    # 加载训练数据
    train_src_path = os.path.join(data_path, f'train.tags.{language_pair}.{src_lang}')
    train_tgt_path = os.path.join(data_path, f'train.tags.{language_pair}.{tgt_lang}')
    
    # 加载验证数据
    # val_src_path = os.path.join(data_path, f'IWSLT17.TED.dev2010.{language_pair}.{src_lang}.xml')
    # val_tgt_path = os.path.join(data_path, f'IWSLT17.TED.dev2010.{language_pair}.{tgt_lang}.xml')
    
    # 加载测试数据（合并多个年份的测试集）
    test_src = []
    test_tgt = []
    test_years = ["2010", "2011", "2012", "2013", "2014", "2015"]
    for y in test_years:
        test_src_path = os.path.join(data_path, f'IWSLT17.TED.tst{y}.{language_pair}.{src_lang}.xml')
        test_tgt_path = os.path.join(data_path, f'IWSLT17.TED.tst{y}.{language_pair}.{tgt_lang}.xml')
        if os.path.isfile(test_src_path) and os.path.isfile(test_tgt_path):
            cur_src = parse_xml_file(test_src_path)
            cur_tgt = parse_xml_file(test_tgt_path)
            # 仅在源/目标句数一致时合并，避免错位
            if len(cur_src) == len(cur_tgt):
                test_src.extend(cur_src)
                test_tgt.extend(cur_tgt)
                print(f"合并测试集: tst{y} | src: {len(cur_src)}, tgt: {len(cur_tgt)}")
            else:
                print(f"警告: tst{y} 源/目标长度不一致，跳过该文件对。src={len(cur_src)} tgt={len(cur_tgt)}")
        else:
            print(f"提示: 缺少测试文件 tst{y}，已跳过。")
    
    # 解析训练数据
    train_src = []
    train_tgt = []
    
    with open(train_src_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # 跳过XML标签行
            if not line.startswith('<'):
                train_src.append(line.strip())
    
    with open(train_tgt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # 跳过XML标签行
            if not line.startswith('<'):
                train_tgt.append(line.strip())
    
    # 打印源语言和目标语言数据长度
    print(f"源语言数据长度: {len(train_src)}, 目标语言数据长度: {len(train_tgt)}")
    
    sentence_pairs = list(zip(train_src, train_tgt))
    # sentence_pairs = sentence_pairs[:50000]
    # 3. 随机打乱数据（非常重要！）
    # 设置一个固定的随机种子，可以保证每次运行的划分结果都一样，便于复现实验
    random.seed(8) 
    random.shuffle(sentence_pairs)

    # 4. 计算分割点
    total_samples = len(sentence_pairs)
    split_index = int(total_samples * 0.8) # 80% 作为训练集

    # 5. 分割数据
    train_pairs = sentence_pairs[:split_index]
    val_pairs = sentence_pairs[split_index:] # 剩下的 20% 作为验证集

    # 6. 将句子对拆分回独立的源和目标列表
    train_src_split, train_tgt_split = zip(*train_pairs)
    val_src_split, val_tgt_split = zip(*val_pairs)

    # 7. 转换回 list 类型 (可选，但通常这么做)
    train_src = list(train_src_split)
    train_tgt = list(train_tgt_split)
    val_src = list(val_src_split)
    val_tgt = list(val_tgt_split)


    # 8. 打印结果，验证一下
    print(f"总样本数: {total_samples}")
    print("-" * 30)
    print(f"划分后的训练集样本数: {len(train_src)}")
    print(f"划分后的验证集样本数: {len(val_src)}")
    print("-" * 30)
    # 解析验证数据
    # val_src = parse_xml_file(val_src_path)
    # val_tgt = parse_xml_file(val_tgt_path)
    
    # 限制样本数量（用于调试）：仅限制训练/验证集，测试集保持完整
    if max_samples:
        train_src = train_src[:max_samples]
        train_tgt = train_tgt[:max_samples]
        val_src = val_src[:max_samples//10]
        val_tgt = val_tgt[:max_samples//10]
    
    # 构建词汇表
    src_vocab = build_vocab(train_src)
    tgt_vocab = build_vocab(train_tgt)
    
    return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt, src_vocab, tgt_vocab

def parse_xml_file(file_path):
    """
    解析XML格式的数据文件
    
    Args:
        file_path: XML文件路径
        
    Returns:
        sentences: 句子列表
    """
    sentences = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # 查找所有文本内容
        for doc in root.findall('.//doc'):
            for seg in doc.findall('.//seg'):
                if seg.text:
                    sentences.append(seg.text.strip())
    except Exception as e:
        print(f"解析XML文件出错: {e}")
    
    return sentences

def build_vocab(texts, min_freq=2, max_size=50000):
    """
    构建词汇表
    
    Args:
        texts: 文本列表
        min_freq: 最小词频
        max_size: 最大词汇表大小
        
    Returns:
        vocab: 词汇表字典，将单词映射到索引
    """
    # 统计词频
    counter = Counter()
    for text in texts:
        words = text.strip().lower().split()
        counter.update(words)
    
    # 过滤低频词并限制词汇表大小
    word_counts = [(w, c) for w, c in counter.items() if c >= min_freq]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    if max_size:
        word_counts = word_counts[:max_size]
    
    # 构建词汇表
    vocab = {}
    vocab['<pad>'] = 0  # 填充符号
    vocab['<unk>'] = 1  # 未知词符号
    vocab['<sos>'] = 2  # 句子开始符号
    vocab['<eos>'] = 3  # 句子结束符号
    
    for i, (word, _) in enumerate(word_counts):
        vocab[word] = i + 4
    
    return vocab

def translate_sentence(model, src_tensor, src_vocab, tgt_vocab, max_len, device):
    """
    使用贪婪解码翻译句子

    Args:
        model: 训练好的Transformer模型
        src_tensor: 源语言句子的张量 [1, src_len]
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        max_len: 最大生成长度
        device: 设备

    Returns:
        translated_sentence: 翻译后的句子
    """
    model.eval()
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    tgt_indexes = [tgt_vocab['<sos>']]

    for i in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indexes).unsqueeze(0).to(device)
        tgt_mask = model.make_tgt_mask(tgt_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(tgt_tensor, enc_src, tgt_mask, src_mask)
        
        pred_token = output.argmax(2)[:, -1].item()
        tgt_indexes.append(pred_token)

        if pred_token == tgt_vocab['<eos>']:
            break

    tgt_tokens = [list(tgt_vocab.keys())[list(tgt_vocab.values()).index(i)] for i in tgt_indexes]
    return " ".join(tgt_tokens[1:-1])




# def create_iwslt_dataloaders(train_src, train_tgt, val_src, val_tgt, src_vocab, tgt_vocab, 
#                             batch_size=32, max_len=100):
def create_iwslt_dataloaders(train_src, train_tgt, val_src, val_tgt, src_vocab, tgt_vocab,
                            batch_size=32, max_len=100, distributed=False, num_workers=0, pin_memory=True):
    """
    为IWSLT2017数据集创建数据加载器
    
    Args:
        train_src: 训练集源语言文本列表
        train_tgt: 训练集目标语言文本列表
        val_src: 验证集源语言文本列表
        val_tgt: 验证集目标语言文本列表
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        batch_size: 批次大小
        max_len: 最大序列长度
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 创建训练集和验证集
    train_dataset = IWSLT2017Dataset(train_src, train_tgt, src_vocab, tgt_vocab, max_len)
    val_dataset = IWSLT2017Dataset(val_src, val_tgt, src_vocab, tgt_vocab, max_len)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建数据加载器（支持分布式采样）
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                                num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
        
    return train_loader, val_loader
    
    # 创建数据集
    train_dataset = SimpleDataset(src_data[train_indices], tgt_data[train_indices])
    val_dataset = SimpleDataset(src_data[val_indices], tgt_data[val_indices])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
    val_dataset = SimpleDataset(src_data[val_indices], tgt_data[val_indices])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader