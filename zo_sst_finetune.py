import argparse
import math
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.optim import AdamW


def prepare_sst2(tokenizer, batch_size=32, max_length=128):
    dataset = load_dataset("glue", "sst2")

    def preprocess(example):
        return tokenizer(
            example["sentence"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    tokenized = dataset.map(preprocess, batched=False)
    tokenized = tokenized.remove_columns([c for c in tokenized["train"].column_names if c not in ["input_ids", "attention_mask", "label"]])

    def collate_fn(features):
        batch = {}
        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f["input_ids"], dtype=torch.long) for f in features], batch_first=True, padding_value=tokenizer.pad_token_id
        )
        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f.get("attention_mask", [1]*len(f["input_ids"])), dtype=torch.long) for f in features], batch_first=True, padding_value=0
        )
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return batch

    train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(tokenized["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


@torch.no_grad()
def zo_gradient_estimator_cls(model, trainable_params, loss_fn, inputs, q, epsilon):
    was_training = model.training
    model.eval()

    original = [p.data.clone() for p in trainable_params]

    def f_loss():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]) 
        logits = outputs.logits
        return loss_fn(logits, inputs["labels"]) 

    seeds, proj_grads = [], []
    for _ in range(q):
        seed = torch.randint(0, 2**31-1, ()).item()
        seeds.append(seed)

        torch.manual_seed(seed)
        for p in trainable_params:
            z = torch.randn_like(p.data)
            p.data.add_(epsilon * z)
        loss_pos = f_loss()

        torch.manual_seed(seed)
        for p, p0 in zip(trainable_params, original):
            z = torch.randn_like(p.data)
            p.data.add_(-2 * epsilon * z)
        loss_neg = f_loss()

        for p, p0 in zip(trainable_params, original):
            p.data.copy_(p0)

        proj_grad = ((loss_pos - loss_neg) / (2 * epsilon)).item()
        proj_grads.append(proj_grad)
        
        # 调试信息：打印前几个投影梯度的值
        if len(proj_grads) <= 3:
            print(f"  Projected grad {len(proj_grads)}: {proj_grad:.6f}, loss_pos: {loss_pos:.6f}, loss_neg: {loss_neg:.6f}")

    grads = [torch.zeros_like(p.data) for p in trainable_params]
    denom = float(len(proj_grads))
    for seed, g in zip(seeds, proj_grads):
        torch.manual_seed(seed)
        for gi, p in enumerate(trainable_params):
            z = torch.randn_like(p.data)
            grads[gi].add_(g * z)
    for gi in range(len(grads)):
        grads[gi].div_(denom)

    if was_training:
        model.train()

    return grads


def train_sst2_zo(model_name, mode, q, lr, epsilon, epochs, batch_size, device, freeze_encoder=True, csv_file=None, log_interval=10):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    train_loader, val_loader = prepare_sst2(tokenizer, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    if tokenizer.pad_token_id is not None:
        model.resize_token_embeddings(len(tokenizer))

    # freeze encoder if requested
    if freeze_encoder:
        for name, p in model.named_parameters():
            if "classifier" not in name and "score" not in name and "lm_head" not in name:
                p.requires_grad = False

    # pick trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(trainable_params, lr=lr) if mode == "FO" else None
    loss_fn = CrossEntropyLoss()

    # csv logging
    if csv_file:
        csv_path = Path(csv_file)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w") as f:
            f.write("timestamp,epoch,step,mode,q,lr,bs,loss,grad_norm,acc\n")

    def eval_acc():
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                logits = model(**{k: batch[k] for k in ("input_ids", "attention_mask")}).logits
                preds = logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].numel()
        model.train()
        return correct / max(1, total)

    step = 0
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            for k in batch:
                batch[k] = batch[k].to(device)

            if mode == "FO":
                optimizer.zero_grad()
                logits = model(**{k: batch[k] for k in ("input_ids", "attention_mask")}).logits
                loss = loss_fn(logits, batch["labels"]) 
                loss.backward()
                optimizer.step()
                grad_norm = 0.0
            else:
                with torch.no_grad():
                    logits = model(**{k: batch[k] for k in ("input_ids", "attention_mask")}).logits
                    loss = loss_fn(logits, batch["labels"]) 

                grads = zo_gradient_estimator_cls(model, trainable_params, loss_fn, batch, q, epsilon)

                # grad norm
                gn2 = 0.0
                for g in grads:
                    gn2 += float(torch.sum(g*g).item())
                grad_norm = math.sqrt(gn2)

                # apply update (SGD) with weight decay for non-bias parameters
                for p, g in zip(trainable_params, grads):
                    if g is None:
                        continue
                    # 简单的权重衰减：只对非bias参数应用
                    if p.dim() > 1:  # 权重矩阵
                        p.data.add_(-lr, g + 0.01 * p.data)
                    else:  # bias向量
                        p.data.add_(-lr, g)

            # 更新进度条显示
            if mode == 'ZO':
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "grad_norm": f"{grad_norm:.4f}",
                    "q": f"{q}"
                })
            else:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}"
                })

            if csv_file and (step % log_interval == 0):
                acc = eval_acc()
                with open(csv_file, "a") as f:
                    f.write(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{epoch+1},{step},{mode},{q if mode=='ZO' else 'N/A'},{lr},{batch_size},{loss.item():.6f},{grad_norm:.6f},{acc:.4f}\n"
                    )
                # 在进度条中显示当前准确率
                current_postfix = pbar.postfix if isinstance(pbar.postfix, dict) else {}
                pbar.set_postfix({
                    **current_postfix,
                    "val_acc": f"{acc:.4f}"
                })

            step += 1

    # final eval
    final_acc = eval_acc()
    print(f"Final validation accuracy: {final_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="SST-2 finetune with ZO/FO")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--mode", type=str, choices=["FO", "ZO"], default="ZO")
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--csv_file", type=str, default="results/sst2_zo.csv")
    parser.add_argument("--log_interval", type=int, default=20)
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    train_sst2_zo(
        model_name=args.model,
        mode=args.mode,
        q=args.q,
        lr=args.lr,
        epsilon=args.epsilon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        freeze_encoder=args.freeze_encoder,
        csv_file=args.csv_file,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()


