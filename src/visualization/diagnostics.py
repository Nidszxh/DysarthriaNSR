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
        object.__setattr__(self, "results_dir", self.root / "results" / "figures" / "data_diagnostics")

def run_diagnostics():
    paths = DataPaths()
    manifest_path = paths.processed_dir / "torgo_neuro_symbolic_manifest.csv"
    
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    missing_phonemes_per_sec = df['phonemes_per_sec'].isna().sum() if 'phonemes_per_sec' in df.columns else len(df)
    missing_manner = df['manner_classes'].isna().sum() if 'manner_classes' in df.columns else len(df)

    print("\nData Cleaning Summary")
    print(f"  Total samples:           {len(df)}")
    print(f"  Missing phonemes_per_sec: {missing_phonemes_per_sec}")
    print(f"  Missing manner_classes:   {missing_manner}")
    
    # Set the style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
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

    # 5. Phonemes per Second (Articulation Rate)
    if 'phonemes_per_sec' in df.columns:
        sns.histplot(ax=axes[0, 2], data=df, x='phonemes_per_sec', hue='status', kde=True, element="step")
        axes[0, 2].set_title("Articulation Rate (Phonemes/Sec)", fontsize=14)
        axes[0, 2].set_xlabel("Phonemes per Second")
    else:
        axes[0, 2].axis('off')
        axes[0, 2].set_title("Articulation Rate (Missing)", fontsize=14)

    # 6. Manner of Articulation Distribution
    if 'manner_classes' in df.columns:
        all_manners = df.assign(
            manner=df['manner_classes'].fillna("").str.split(' ')
        ).explode('manner')
        all_manners = all_manners[all_manners['manner'].str.strip() != ""]
        sns.countplot(ax=axes[1, 2], data=all_manners, x='manner', hue='status', palette='tab10')
        axes[1, 2].set_title("Distribution of Manner of Articulation", fontsize=14)
        axes[1, 2].tick_params(axis='x', rotation=45)
    else:
        axes[1, 2].axis('off')
        axes[1, 2].set_title("Manner Distribution (Missing)", fontsize=14)

    plt.tight_layout()
    save_path = paths.results_dir / "dataset_diagnostics.png"
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    run_diagnostics()