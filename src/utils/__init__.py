import re
from pathlib import Path
from typing import Optional


def resolve_best_fold_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Return checkpoint with lowest val_per in filename for this fold.

    Falls back to last.ckpt when no score-tagged checkpoints exist.
    """
    if not ckpt_dir.exists():
        return None

    scored = list(ckpt_dir.glob("epoch=*-val_per=*.ckpt"))
    if scored:
        pattern = re.compile(r"val_per=([0-9]+(?:\.[0-9]+)?)")
        best = None
        best_score = float("inf")
        for p in scored:
            m = pattern.search(p.name)
            if not m:
                continue
            score = float(m.group(1))
            if score < best_score:
                best_score = score
                best = p
        if best is not None:
            return best

    last_ckpt = ckpt_dir / "last.ckpt"
    return last_ckpt if last_ckpt.exists() else None
