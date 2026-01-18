import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass, field

@dataclass(frozen=True)
class DataPaths:
    root: Path = Path(__file__).resolve().parents[2]
    processed_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    
    def __post_init__(self):
        object.__setattr__(self, "processed_dir", self.root / "data" / "processed")
        object.__setattr__(self, "results_dir", self.root / "results" / "figures")

def run_diagnostics():
    paths = DataPaths()
    manifest_path = paths.processed_dir / "torgo_neuro_symbolic_manifest.csv"
    
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Class Balance
    sns.countplot(ax=axes[0, 0], data=df, x='status', hue='status', palette='viridis', legend=False)
    axes[0, 0].set_title("Distribution of Speech Status", fontsize=14)
    
    # 2. Duration vs Status (Crucial for VRAM planning)
    sns.histplot(ax=axes[0, 1], data=df, x='duration', hue='status', kde=True, element="step")
    axes[0, 1].set_title("Audio Duration Distribution", fontsize=14)
    axes[0, 1].set_xlabel("Seconds")

    # 3. Phoneme Complexity
    sns.boxplot(ax=axes[1, 0], data=df, x='status', y='phn_count', palette='magma', hue='status', legend=False)
    axes[1, 0].set_title("Phoneme Count per Utterance", fontsize=14)
    
    # 4. Speaker Contribution (Check for data leakage/bias)
    speaker_counts = df['speaker'].value_counts()
    speaker_counts.plot(kind='bar', ax=axes[1, 1], color='skyblue')
    axes[1, 1].set_title("Samples per Speaker", fontsize=14)
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_path = paths.results_dir / "dataset_diagnostics.png"
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    run_diagnostics()