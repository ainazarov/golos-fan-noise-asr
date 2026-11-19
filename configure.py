"""Train the text-only tokenizer and write explicit experiment configurations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sentencepiece as spm
from omegaconf import OmegaConf, open_dict

SOURCE_CONFIG = Path("/share/asr/Speech/examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_tdt_ctc_bpe_ru.yaml")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    root = args.output_root.resolve()
    tokenizer = root / "tokenizer112"
    tokenizer.mkdir(exist_ok=True)
    model_file = tokenizer / "tokenizer.model"
    if model_file.exists() and not args.overwrite:
        raise FileExistsError(f"{model_file}; use --overwrite explicitly")
    spm.SentencePieceTrainer.train(
        input=str(root / "data" / "train_text.txt"), model_prefix=str(tokenizer / "tokenizer"),
        model_type="bpe", vocab_size=112, character_coverage=1.0,
        bos_id=-1, eos_id=-1, pad_id=-1, unk_id=0, max_sentencepiece_length=4,
        normalization_rule_name="identity", remove_extra_whitespaces=False,
        shuffle_input_sentence=False, input_sentence_size=0, num_threads=1,
        hard_vocab_limit=True,
    )
    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    assert sp.get_piece_size() == 112
    (tokenizer / "vocab.txt").write_text("\n".join(sp.id_to_piece(i) for i in range(1, 112)) + "\n", encoding="utf-8")
    (root / "configs").mkdir(exist_ok=True)
    base = OmegaConf.load(args.source_config)
    with open_dict(base):
        base.pop("exp_manager", None)
        base.name = "golos112_fan_from_scratch"
        base.seed = 112
        base.model.train_ds.manifest_filepath = str(root / "data" / "train.jsonl")
        base.model.validation_ds.manifest_filepath = str(root / "data" / "val_original.jsonl")
        base.model.test_ds.manifest_filepath = None
        base.model.tokenizer.dir = str(tokenizer)
        base.model.tokenizer.type = "bpe"
        base.model.preprocessor.features = 128
        base.model.encoder.feat_in = 128
        base.model.train_ds.use_lhotse = False
        base.model.validation_ds.use_lhotse = False
        base.model.train_ds.batch_size = 16
        base.model.validation_ds.batch_size = 16
        base.model.train_ds.num_workers = 8
        base.model.validation_ds.num_workers = 4
        base.model.log_prediction = False
        base.model.aux_ctc.decoding.strategy = "greedy_batch"
        base.model.optim.lr = 0.0003
        base.model.optim.sched = {"name": "CosineAnnealing", "warmup_steps": 5000, "min_lr": 1e-6}
        base.trainer = {"devices": 1, "accelerator": "gpu", "max_epochs": 10,
                        "precision": 32, "gradient_clip_val": 1.0,
                        "accumulate_grad_batches": 1, "log_every_n_steps": 50,
                        "num_sanity_val_steps": 0, "check_val_every_n_epoch": 5,
                        "enable_progress_bar": False, "benchmark": False}
    for phase in ["baseline", "mixed", "noise"]:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        if phase != "baseline":
            cfg.trainer.max_epochs = 25 if phase == "mixed" else 5
            cfg.model.optim.lr = 0.0001 if phase == "mixed" else 0.00003
            cfg.model.optim.sched.warmup_steps = 1000
            cfg.model.train_ds.manifest_filepath = str(root / "data" / "train_mixed.jsonl")
            cfg.model.validation_ds.manifest_filepath = str(
                root / "data" / ("val_mixed_original.jsonl" if phase == "mixed" else "val_mixed_selection.jsonl"))
        if phase == "noise":
            cfg.model.train_ds.augmentor = {"proj1_fan_noise": {
                "prob": 0.5, "manifest_path": str(root / "data" / "noise_train.jsonl"),
                "min_snr_db": 5.0, "max_snr_db": 20.0}}
        destination = root / "configs" / (phase + ".yaml")
        if destination.exists() and not args.overwrite:
            raise FileExistsError(destination)
        OmegaConf.save(cfg, destination)
    report = {"vocab_size": sp.get_piece_size(), "unknown_id": sp.unk_id(),
              "tokenizer_training_text": "data/train_text.txt (training split only)",
              "max_piece_length": 4, "sample": sp.encode("привет мир", out_type=str),
              "source_config": str(args.source_config), "mel_bins": 128, "baseline_epochs": 10,
              "mixed_epochs": 25, "noise_epochs": 5, "validation_every_epochs": 5,
              "checkpoint_every_epochs": 1, "full_training_started": False}
    (root / "reports" / "configuration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
