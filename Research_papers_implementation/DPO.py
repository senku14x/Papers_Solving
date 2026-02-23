import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class DPOConfig:
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    reference_free: bool = False
    lr: float = 1e-6
    batch_size: int = 4
    max_steps: int = 1000
    grad_clip: float = 1.0
    warmup_steps: int = 100
    log_every: int = 50


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class ToyLM(nn.Module):
    def __init__(self, vocab_size=256, d_model=128, n_heads=4, n_layers=2, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.ln_f(x))

    def log_probs_from_logits(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        shift_logits = log_probs[:, :-1, :]
        shift_labels = labels[:, 1:]
        token_lp = shift_logits.gather(
            dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)
        mask = (shift_labels != -100).float()
        return (token_lp * mask).sum(dim=-1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, cfg):
    if cfg.reference_free:
        ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
        ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

    log_ratio_w = policy_chosen_logps - ref_chosen_logps
    log_ratio_l = policy_rejected_logps - ref_rejected_logps
    reward_w = cfg.beta * log_ratio_w
    reward_l = cfg.beta * log_ratio_l
    reward_margin = reward_w - reward_l

    if cfg.loss_type == "sigmoid":
        loss = (
            -F.logsigmoid(reward_margin) * (1 - cfg.label_smoothing)
            - F.logsigmoid(-reward_margin) * cfg.label_smoothing
        )
    elif cfg.loss_type == "hinge":
        loss = F.relu(1 - reward_margin)
    elif cfg.loss_type == "ipo":
        loss = (reward_margin / cfg.beta - 1 / (2 * cfg.beta)) ** 2
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type}")

    loss = loss.mean()
    metrics = {
        "loss": loss.item(),
        "reward_chosen": reward_w.mean().item(),
        "reward_rejected": reward_l.mean().item(),
        "reward_margin": reward_margin.mean().item(),
        "accuracy": (reward_margin > 0).float().mean().item(),
    }
    return loss, metrics


@dataclass
class PreferenceExample:
    prompt: list[int]
    chosen: list[int]
    rejected: list[int]


class PreferenceDataset(Dataset):
    def __init__(self, examples, max_len=128):
        self.examples = examples
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def _make_input_and_labels(self, prompt, response):
        ids = (prompt + response)[:self.max_len]
        labels = ([-100] * len(prompt) + response)[:self.max_len]
        pad_len = self.max_len - len(ids)
        ids += [0] * pad_len
        labels += [-100] * pad_len
        return ids, labels

    def __getitem__(self, idx):
        ex = self.examples[idx]
        c_ids, c_labels = self._make_input_and_labels(ex.prompt, ex.chosen)
        r_ids, r_labels = self._make_input_and_labels(ex.prompt, ex.rejected)
        return {
            "chosen_input_ids": torch.tensor(c_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(c_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(r_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(r_labels, dtype=torch.long),
        }


class DPOTrainer:
    def __init__(self, policy, reference, dataset, cfg, device="cpu"):
        self.policy = policy.to(device)
        self.reference = reference.to(device)
        self.cfg = cfg
        self.device = device

        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.reference.eval()

        self.loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: step / max(1, cfg.warmup_steps) if step < cfg.warmup_steps else 1.0
        )

    @torch.no_grad()
    def _get_logps(self, model, input_ids, labels):
        return model.log_probs_from_logits(model(input_ids), labels)

    def train(self):
        self.policy.train()
        step = 0
        running = {k: 0.0 for k in ["loss", "reward_chosen", "reward_rejected", "reward_margin", "accuracy"]}

        print(f"Starting DPO training | β={self.cfg.beta} | loss={self.cfg.loss_type}")
        print("-" * 65)

        while step < self.cfg.max_steps:
            for batch in self.loader:
                if step >= self.cfg.max_steps:
                    break

                chosen_ids = batch["chosen_input_ids"].to(self.device)
                chosen_lbl = batch["chosen_labels"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                rejected_lbl = batch["rejected_labels"].to(self.device)

                ref_chosen_lp = self._get_logps(self.reference, chosen_ids, chosen_lbl)
                ref_rejected_lp = self._get_logps(self.reference, rejected_ids, rejected_lbl)

                pol_chosen_lp = self.policy.log_probs_from_logits(self.policy(chosen_ids), chosen_lbl)
                pol_rejected_lp = self.policy.log_probs_from_logits(self.policy(rejected_ids), rejected_lbl)

                loss, metrics = dpo_loss(pol_chosen_lp, pol_rejected_lp, ref_chosen_lp, ref_rejected_lp, self.cfg)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                step += 1

                for k in running:
                    running[k] += metrics[k]

                if step % self.cfg.log_every == 0:
                    avg = {k: v / self.cfg.log_every for k, v in running.items()}
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"step {step:5d} | loss={avg['loss']:.4f} | "
                        f"margin={avg['reward_margin']:.4f} | "
                        f"acc={avg['accuracy']:.3f} | "
                        f"r_w={avg['reward_chosen']:.3f} | "
                        f"r_l={avg['reward_rejected']:.3f} | "
                        f"lr={lr:.2e}"
                    )
                    running = {k: 0.0 for k in running}

        print("-" * 65)
        print("Training complete.")


def make_synthetic_dataset(n=512, vocab_size=256, prompt_len=16, resp_len=24):
    examples = []
    for _ in range(n):
        prompt = torch.randint(1, vocab_size, (prompt_len,)).tolist()
        chosen = torch.randint(vocab_size // 2, vocab_size, (resp_len,)).tolist()
        rejected = torch.randint(1, vocab_size // 2, (resp_len,)).tolist()
        examples.append(PreferenceExample(prompt, chosen, rejected))
    return examples


if __name__ == "__main__":
    import copy

    VOCAB = 256
    D_MODEL = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    policy = ToyLM(vocab_size=VOCAB, d_model=D_MODEL, n_heads=4, n_layers=2)
    reference = copy.deepcopy(policy)

    dataset = PreferenceDataset(make_synthetic_dataset(n=1024, vocab_size=VOCAB), max_len=64)

    cfg = DPOConfig(
        beta=0.1,
        label_smoothing=0.0,
        loss_type="sigmoid",
        lr=1e-4,
        batch_size=16,
        max_steps=500,
        log_every=50,
    )

    DPOTrainer(policy, reference, dataset, cfg, device).train()
