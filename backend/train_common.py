"""train_common.py — briques partagées par dpo_train / grpo_train / ppo_train / reward_model."""

from __future__ import annotations

import dataclasses
import gc
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dataset_builder import load_holdout_questions


# --------------------------------------------------------------------------- #
# Registries
# --------------------------------------------------------------------------- #

class JsonRegistry:
    """Liste de versions persistée dans <root>/registry.json."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        if not self.index_path.exists():
            self._save([])

    def _load(self) -> list[dict]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save(self, entries: list[dict]) -> None:
        self.index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def history(self) -> list[dict]:
        return self._load()

    def next_version(self) -> int:
        return max((e["version"] for e in self._load()), default=0) + 1

    def register(self, entry) -> None:
        entries = self._load()
        entries.append(dataclasses.asdict(entry))
        self._save(entries)


@dataclass
class ModelVersion:
    version: int
    base_model: str
    created_at: str
    dataset_version: int
    n_training_pairs: int
    path: str
    status: str = "staged"     # staged | active | retired
    method: str = "dpo"        # dpo | ppo | grpo
    eval_summary: dict = field(default_factory=dict)


class ModelRegistry(JsonRegistry):
    def __init__(self, root: str | Path = "models"):
        super().__init__(root)

    def active_version(self) -> dict | None:
        active = [e for e in self._load() if e["status"] == "active"]
        return active[-1] if active else None

    def promote(self, version: int) -> None:
        entries = self._load()
        if not any(e["version"] == version for e in entries):
            raise ValueError(f"Version inconnue : v{version}")
        for e in entries:
            if e["status"] == "active":
                e["status"] = "retired"
            if e["version"] == version:
                e["status"] = "active"
        self._save(entries)

    def set_eval_summary(self, version: int, summary: dict) -> None:
        entries = self._load()
        for e in entries:
            if e["version"] == version:
                e["eval_summary"] = summary
        self._save(entries)


def register_trained(registry: ModelRegistry, *, base_model: str, version: int, path: str,
                     n_pairs: int, method: str, dataset_version: int = 0) -> ModelVersion:
    mv = ModelVersion(
        version=version, base_model=base_model,
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_version=dataset_version, n_training_pairs=n_pairs,
        path=path, status="staged", method=method,
    )
    registry.register(mv)
    return mv


# --------------------------------------------------------------------------- #
# Mémoire / dtype / config
# --------------------------------------------------------------------------- #

def resolve_training_dtype():
    """(dtype, has_cuda, has_bf16). fp32 sur CPU, bf16/fp16 sur GPU (poids 2x plus légers)."""
    import torch
    if not torch.cuda.is_available():
        return torch.float32, False, False
    has_bf16 = torch.cuda.is_bf16_supported()
    return (torch.bfloat16 if has_bf16 else torch.float16), True, has_bf16


def print_gpu_mem(tag: str) -> None:
    import torch
    if torch.cuda.is_available():
        print(f"[mem] {tag}: allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB, "
              f"reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB")


def free_memory(*_ignored) -> None:
    """gc + vidage du cache CUDA. Les objets doivent être `del` par l'appelant
    AVANT cet appel (supprimer un argument ici ne libère rien)."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def lora_config(task_type: str = "CAUSAL_LM"):
    from peft import LoraConfig
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"], task_type=task_type)


def make_config(config_cls, tag: str, **kwargs):
    """Ne passe que les champs que la version de trl installée accepte (trl
    renomme/supprime des champs d'une version à l'autre) et loggue le reste."""
    accepted = {f.name for f in dataclasses.fields(config_cls)}
    dropped = sorted(set(kwargs) - accepted)
    if dropped:
        print(f"[{tag}] ⚠ options ignorées par cette version de trl : {dropped}")
    return config_cls(**{k: v for k, v in kwargs.items() if k in accepted})


def load_merged(auto_cls, path: str, base_model: str, dtype, **kwargs):
    """Charge un modèle complet OU un adaptateur LoRA (fusionné sur son modèle de base)."""
    if (Path(path) / "adapter_config.json").exists():
        from peft import PeftModel
        base = auto_cls.from_pretrained(base_model, torch_dtype=dtype, **kwargs)
        return PeftModel.from_pretrained(base, path).merge_and_unload()
    return auto_cls.from_pretrained(path, torch_dtype=dtype, **kwargs)


def count_usable_pairs(store) -> int:
    """Paires réellement utilisables pour l'entraînement (hors holdout, hors paires dégénérées)."""
    holdout = load_holdout_questions() or set()
    n = 0
    for r in store.validated_corrections():
        chosen = r["final_answer"] or r["proposed_answer"]
        if (r["question"] not in holdout and chosen and r["rejected_answer"]
                and chosen.strip() != r["rejected_answer"].strip()):
            n += 1
    return n