"""
Fine-tuning cardiffnlp/twitter-roberta-base on PHEME
======================================================
目標：
  讓 RoBERTa 學會區分謠言和非謠言的文字風格，
  之後用 fine-tuned 模型抽節點嵌入餵給 GNN。

訓練資料：
  每則推文的文字（source tweet + reactions）
  + 繼承自 thread 的 is_rumour 標籤（0/1）

重要設計原則：
  - LOEO-aware fine-tuning：每個 fold 只用 train 事件的文字做 fine-tuning，
    test 事件的文字完全不碰，避免 event leakage。
  - 但 9 個 fold 各自 fine-tuning 一次很耗時，
    所以這裡提供兩種模式：
      1. full_finetune=True：每個 fold 各自 fine-tuning（最嚴謹）
      2. full_finetune=False：用全部資料 fine-tuning 一次，
         然後用同一個模型抽所有節點的嵌入（速度快，但有輕微 leakage）
    建議先用模式 2 驗證效果，確認有提升後再跑模式 1。

輸出：
  - fine-tuned 模型存在 ./roberta_pheme/ 資料夾
  - 抽好的嵌入存在 pheme_node_features_finetuned.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

MODEL_NAME  = "cardiffnlp/twitter-roberta-base"
MAX_LENGTH  = 128   # Twitter 推文最長 280 字元，128 token 通常足夠
EMBED_DIM   = 768   # RoBERTa hidden size
BATCH_SIZE  = 32
EPOCHS      = 3     # fine-tuning 通常 3-5 epoch 就夠，太多會 overfit
LR          = 2e-5  # RoBERTa fine-tuning 的標準學習率
WARMUP_RATIO = 0.1  # 前 10% steps 做 warmup


# ------------------------------------------------------------------
# 1. Dataset
# ------------------------------------------------------------------

class TweetDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ------------------------------------------------------------------
# 2. Fine-tuning
# ------------------------------------------------------------------

def finetune_roberta(
    train_texts: list[str],
    train_labels: list[int],
    output_dir: str = "./roberta_pheme",
    model_name: str = MODEL_NAME,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    verbose: bool = True,
) -> str:
    """
    Fine-tune RoBERTa 做二元分類（rumour vs non-rumour）。
    回傳 fine-tuned 模型的存放路徑。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Fine-tuning 裝置：{device}")
        print(f"訓練樣本數：{len(train_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    # 切出 10% 做 validation
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42,
        stratify=train_labels,
    )

    tr_dataset  = TweetDataset(tr_texts,  tr_labels,  tokenizer, MAX_LENGTH)
    val_dataset = TweetDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    tr_loader  = DataLoader(tr_dataset,  batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(tr_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += len(labels)

        val_acc = correct / total
        if verbose:
            print(f"  epoch {epoch} | loss={total_loss/len(tr_loader):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 存最佳模型
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if verbose:
        print(f"Fine-tuned 模型已存至 {output_dir}（最佳 val_acc={best_val_acc:.4f}）")

    return output_dir


# ------------------------------------------------------------------
# 3. 抽取嵌入（用 fine-tuned 模型的 [CLS] token）
# ------------------------------------------------------------------

def extract_embeddings(
    texts: list[Optional[str]],
    model_dir: str,
    batch_size: int = 64,
    verbose: bool = True,
) -> np.ndarray:
    """
    用 fine-tuned RoBERTa 抽取每則推文的嵌入（[CLS] token 的 hidden state）。
    text 為 None 或空字串：填零向量。
    回傳 shape: (len(texts), EMBED_DIM)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device)
    model.eval()

    embeddings = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)

    valid_idx   = [
        i for i, t in enumerate(texts)
        if t and isinstance(t, str) and t.strip()
    ]
    valid_texts = [texts[i] for i in valid_idx]

    if verbose:
        print(f"抽取嵌入：{len(valid_texts)} 則有效推文（共 {len(texts)} 則）")

    for start in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # [CLS] token 的 hidden state 作為句子嵌入
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        for j, global_idx in enumerate(valid_idx[start:start + batch_size]):
            embeddings[global_idx] = cls_emb[j]

        if verbose and (start // batch_size + 1) % 10 == 0:
            print(f"  進度：{start + len(batch_texts)}/{len(valid_texts)}")

    return embeddings


# ------------------------------------------------------------------
# 4. 主流程：全資料 fine-tuning（模式 2，速度快）
# ------------------------------------------------------------------

def run_full_finetune_and_extract(
    reply_csv: str = "pheme_reply_level.csv",
    node_feat_csv: str = "pheme_node_features.csv",
    output_csv: str = "pheme_node_features_finetuned.csv",
    model_output_dir: str = "./roberta_pheme",
    verbose: bool = True,
):
    """
    用全部 PHEME 資料 fine-tuning RoBERTa，然後抽取節點嵌入。
    這是模式 2（速度快，有輕微 leakage），適合先驗證效果。
    """
    # 讀取資料
    reply_df = pd.read_csv(
        reply_csv,
        dtype={"tweet_id": str, "parent_id": str, "thread_id": str}
    )
    feat_df = pd.read_csv(
        node_feat_csv,
        dtype={"tweet_id": str, "thread_id": str},
        low_memory=False,
    )

    # 準備 fine-tuning 資料
    # 只用有文字的推文，標籤繼承 thread 的 is_rumour
    valid_mask = reply_df["text"].apply(
        lambda t: isinstance(t, str) and bool(t.strip())
    )
    train_texts  = reply_df.loc[valid_mask, "text"].tolist()
    train_labels = reply_df.loc[valid_mask, "is_rumour"].tolist()

    if verbose:
        print(f"Fine-tuning 樣本：{len(train_texts)} 則推文")
        rumour_count = sum(train_labels)
        print(f"  rumour={rumour_count}, non-rumour={len(train_labels)-rumour_count}")

    # Fine-tuning
    finetune_roberta(
        train_texts, train_labels,
        output_dir=model_output_dir,
        verbose=verbose,
    )

    # 抽取嵌入
    texts = reply_df["text"].tolist()
    embeddings = extract_embeddings(texts, model_output_dir, verbose=verbose)

    # 替換 feat_df 裡的嵌入欄位
    old_emb_cols = [c for c in feat_df.columns if c.startswith("emb_")]
    feat_df = feat_df.drop(columns=old_emb_cols)

    emb_cols = [f"emb_{i}" for i in range(EMBED_DIM)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=feat_df.index)
    feat_df = pd.concat([feat_df, emb_df], axis=1)

    feat_df.to_csv(output_csv, index=False)
    if verbose:
        print(f"已存至 {output_csv}，shape={feat_df.shape}")

    return feat_df


# ------------------------------------------------------------------
# 5. LOEO-aware fine-tuning（模式 1，最嚴謹）
# ------------------------------------------------------------------

def run_loeo_finetune_and_extract(
    reply_csv: str = "pheme_reply_level.csv",
    node_feat_csv: str = "pheme_node_features_roberta.csv",
    output_csv: str = "pheme_node_features_loeo_finetuned.csv",
    model_base_dir: str = "./roberta_pheme_loeo",
    verbose: bool = True,
):
    """
    LOEO-aware fine-tuning：
    每個 fold 只用 train 事件的推文做 fine-tuning，
    test 事件的推文完全不碰，最後合併成完整的 node_features CSV。

    流程：
      for each test_event in 9 events:
        1. train_texts = 其他 8 個事件的推文
        2. fine-tuning RoBERTa
        3. 抽取 test_event 的節點嵌入
        4. 存暫存檔
      合併 9 個 fold 的嵌入 → 輸出完整 CSV

    注意：
      - 每個 fold 的模型存在 {model_base_dir}/{test_event}/
      - 如果某個 fold 的模型已存在，直接跳過 fine-tuning（斷點續跑）
    """
    reply_df = pd.read_csv(
        reply_csv,
        dtype={"tweet_id": str, "parent_id": str, "thread_id": str}
    )
    feat_df = pd.read_csv(
        node_feat_csv,
        dtype={"tweet_id": str, "thread_id": str},
        low_memory=False,
    )

    events = sorted(reply_df["event_id"].unique())
    if verbose:
        print(f"事件列表：{events}")

    # 收集每個 fold 的嵌入，最後合併
    # 用 dict 存 {tweet_id: embedding}，確保順序對齊
    all_embeddings = np.zeros((len(reply_df), EMBED_DIM), dtype=np.float32)

    for test_event in events:
        if verbose:
            print(f"\n{'='*55}")
            print(f"LOEO fold：test_event = {test_event}")

        # 分出 train / test 的推文列
        train_mask = reply_df["event_id"] != test_event
        test_mask  = reply_df["event_id"] == test_event

        train_reply = reply_df[train_mask]
        test_reply  = reply_df[test_mask]

        # Fine-tuning 資料（只用 train 事件）
        valid_train = train_reply["text"].apply(
            lambda t: isinstance(t, str) and bool(t.strip())
        )
        train_texts  = train_reply.loc[valid_train, "text"].tolist()
        train_labels = train_reply.loc[valid_train, "is_rumour"].tolist()

        if verbose:
            print(f"  train 推文數：{len(train_texts)}，test 推文數：{len(test_reply)}")

        # Fine-tuning（若已存在則跳過，支援斷點續跑）
        model_dir = str(Path(model_base_dir) / test_event)
        if Path(model_dir).exists() and (Path(model_dir) / "config.json").exists():
            if verbose:
                print(f"  找到已存在的模型，跳過 fine-tuning：{model_dir}")
        else:
            finetune_roberta(
                train_texts, train_labels,
                output_dir=model_dir,
                verbose=verbose,
            )

        # 抽取 test_event 的嵌入
        test_texts = test_reply["text"].tolist()
        test_embs  = extract_embeddings(test_texts, model_dir, verbose=verbose)

        # 把嵌入填回對應的行
        test_indices = test_reply.index.tolist()
        for local_i, global_i in enumerate(test_indices):
            all_embeddings[global_i] = test_embs[local_i]

    # 替換 feat_df 裡的嵌入欄位
    old_emb_cols = [c for c in feat_df.columns if c.startswith("emb_")]
    feat_df = feat_df.drop(columns=old_emb_cols)

    emb_cols = [f"emb_{i}" for i in range(EMBED_DIM)]
    emb_df = pd.DataFrame(all_embeddings, columns=emb_cols, index=feat_df.index)
    feat_df = pd.concat([feat_df, emb_df], axis=1)

    feat_df.to_csv(output_csv, index=False)
    if verbose:
        print(f"\n完成！已存至 {output_csv}，shape={feat_df.shape}")

    return feat_df


# ------------------------------------------------------------------
# 6. 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # 預設跑 LOEO-aware 版本（模式 1）
    # 如果想跑快速版本（模式 2），傳入參數 --fast
    if "--fast" in sys.argv:
        print("跑模式 2（快速版，有 event leakage，僅供驗證）")
        run_full_finetune_and_extract(
            reply_csv       = "pheme_reply_level.csv",
            node_feat_csv   = "pheme_node_features_roberta.csv",
            output_csv      = "pheme_node_features_finetuned_fast.csv",
            model_output_dir= "./roberta_pheme_full",
        )
    else:
        print("跑模式 1（LOEO-aware，最嚴謹，約 2-3 小時）")
        run_loeo_finetune_and_extract(
            reply_csv       = "pheme_reply_level.csv",
            node_feat_csv   = "pheme_node_features_roberta.csv",
            output_csv      = "pheme_node_features_loeo_finetuned.csv",
            model_base_dir  = "./roberta_pheme_loeo",
        )