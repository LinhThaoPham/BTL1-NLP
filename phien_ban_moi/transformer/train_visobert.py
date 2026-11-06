"""Fine-tune ViSoBERT (uitnlp/visobert) trên bộ chia hiện có, 2 kịch bản như NB/SVM.

Chạy tại thư mục BTL1-NLP:
    .venv\\Scripts\\python.exe -m phien_ban_moi.transformer.train_visobert

Quy trình giữ giống BTL1: chọn epoch và kịch bản bằng Macro-F1 validation,
ghi lựa chọn ra file trước khi chạy test; ROS chỉ lặp mẫu trong train.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..evaluation.metrics import evaluate
from ..training.sampling import random_oversample_indices
from .data import BASE_DIR, TEXT_COLUMN, Encoded, load_splits, make_collate

SCENARIOS = {
    "original_imbalanced": "A — gốc, chưa cân bằng",
    "original_ros": "B — gốc + Random Oversampling",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict_scores(model, dataset, collate, device, batch_size):
    """Trả về xác suất lớp Toxic theo đúng thứ tự dataset."""
    import torch
    from torch.utils.data import DataLoader
    model.eval()
    scores = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            scores.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def train_scenario(scenario, encoded, collate, args, output, device):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    set_seed(args.seed)
    train, validation = encoded["train"], encoded["validation"]
    y = train.labels
    order = np.arange(len(y)) if scenario == "original_imbalanced" else random_oversample_indices(y, args.seed)
    pd.DataFrame({"train_index": order, "label": y[order]}).to_csv(
        output / "data" / f"{scenario}_sampling.csv", index=False)
    subset = torch.utils.data.Subset(train, order.tolist())
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=generator)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label={0: "Non Toxic", 1: "Toxic"}, label2id={"Non Toxic": 0, "Toxic": 1}
    ).to(device)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = math.ceil(len(loader) / args.accumulation)
    total = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_f1, best_epoch, history = -1.0, None, []
    model_dir = output / "models" / scenario
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, batches = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**batch).loss / args.accumulation
            scaler.scale(loss).backward()
            running += loss.item() * args.accumulation
            batches += 1
            if step % args.accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        val_scores = predict_scores(model, validation, collate, device, args.eval_batch_size)
        val_pred = (val_scores >= 0.5).astype(int)
        val_f1 = f1_score(validation.labels, val_pred, average="macro", zero_division=0)
        history.append({"scenario": scenario, "epoch": epoch, "train_loss": running / batches,
                        "validation_macro_f1": val_f1,
                        "validation_toxic_f1": f1_score(validation.labels, val_pred, pos_label=1, zero_division=0),
                        "elapsed_seconds": time.perf_counter() - started})
        print(f"{scenario} epoch {epoch}: loss={running / batches:.4f} val Macro-F1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            model.save_pretrained(model_dir)
    del model, optimizer
    torch.cuda.empty_cache()
    return {"model": f"{scenario}__ViSoBERT", "scenario": scenario, "feature": "ViSoBERT", "algorithm": "fine-tune",
            "validation_macro_f1": best_f1, "best_epoch": best_epoch, "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size * args.accumulation, "max_length": args.max_length,
            "fit_seconds": time.perf_counter() - started, "train_before_ros": len(y),
            "train_after_ros": len(order)}, history


def run(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    output = Path(args.output) if args.output else BASE_DIR / "results" / ("visobert_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    for name in ("data", "models", "predictions"):
        (output / name).mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiết bị:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU", flush=True)
    pieces = load_splits(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(output / "models" / "tokenizer")
    encoded = {name: Encoded(part[TEXT_COLUMN], part.label, tokenizer, args.max_length) for name, part in pieces.items()}
    collate = make_collate(tokenizer.pad_token_id)
    lengths = {name: {"rows": len(enc), "truncated_at_max_length": enc.truncated,
                      "median_tokens": float(np.median([len(i) for i in enc.input_ids]))} for name, enc in encoded.items()}
    write_json(output / "token_stats.json", lengths)
    print("Token:", lengths, flush=True)

    choices, histories = [], []
    for scenario in args.scenarios:
        choice, history = train_scenario(scenario, encoded, collate, args, output, device)
        choices.append(choice)
        histories.extend(history)
    choices = pd.DataFrame(choices)
    choices.to_csv(output / "validation_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(histories).to_csv(output / "training_history.csv", index=False, encoding="utf-8-sig")
    selected = choices.loc[choices.validation_macro_f1.idxmax(), "model"]
    write_json(output / "selected_model_before_test.json",
               {"selected_model": selected, "criterion": "validation_macro_f1", "test_used_for_selection": False})

    # Lựa chọn đã chốt; từ đây mới đọc test.
    test = pieces["test"]
    collected = ~test.is_synthetic.to_numpy()
    records = []
    for choice in choices.to_dict("records"):
        model = AutoModelForSequenceClassification.from_pretrained(output / "models" / choice["scenario"]).to(device)
        scores = predict_scores(model, encoded["test"], collate, device, args.eval_batch_size)
        predicted = (scores >= 0.5).astype(int)
        measures, report, matrix = evaluate(test.label, predicted, scores)
        record = {**choice, **{"test_" + k: v for k, v in measures.items()}}
        subset, _, sub_matrix = evaluate(test.label[collected], predicted[collected], scores[collected])
        record.update({"collected_test_" + k: v for k, v in subset.items()})
        records.append(record)
        name = choice["model"]
        write_json(output / "predictions" / f"{name}_confusion.json", matrix.tolist())
        write_json(output / "predictions" / f"{name}_collected_confusion.json", sub_matrix.tolist())
        write_json(output / "predictions" / f"{name}_classification_report.json", report)
        test[["record_id", TEXT_COLUMN, "label", "is_synthetic", "post_id"]].assign(
            predicted_label=predicted, score_toxic=scores
        ).to_csv(output / "predictions" / f"{name}.csv", index=False, encoding="utf-8-sig")
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    from importlib.metadata import version
    write_json(output / "metadata.json", {
        "model_name": args.model, "text_column": TEXT_COLUMN, "data_dir": str(Path(args.data_dir).resolve()),
        "seed": args.seed, "selected_model": selected, "decision_threshold": 0.5,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "versions": {p: version(p) for p in ("torch", "transformers", "scikit-learn", "pandas")},
        "args": vars(args)})
    write_json(output / "run_status.json", {"status": "complete", "selected_model": selected})
    write_json(BASE_DIR / "results" / "latest_visobert.json", {"run_dir": str(output.resolve())})
    print(pd.DataFrame(records)[["model", "validation_macro_f1", "best_epoch", "test_macro_f1", "test_toxic_f1",
                                 "collected_test_macro_f1", "collected_test_toxic_f1"]].to_string(index=False))
    print("HOÀN THÀNH:", output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data" / "processed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="uitnlp/visobert")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
