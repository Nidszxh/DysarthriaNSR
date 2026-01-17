## High-Level Project Overview

![Image](https://www.researchgate.net/publication/346664731/figure/fig1/AS%3A965899142041602%401607299709195/A-Conventional-Speech-Recognition-System-Pipeline-Described-by-Schalkwyk-11.ppm)

![Image](https://neurosys.com/wp-content/uploads/2021/09/1_training-phases-of-wav2vec-2.0-1024x574.png)

![Image](https://maelfabien.github.io/assets/images/asr_11.png)

![Image](https://maelfabien.github.io/assets/images/asr_12.png)

### Project Goal

The objective of this project is to design a **robust and explainable dysarthric speech recognition system** by combining **self-supervised neural speech representations** with **symbolic phoneme-level reasoning**. The system aims to overcome the limitations of conventional ASR models when dealing with impaired speech, while also providing **interpretable explanations** for its predictions.

---

### Core Idea

Traditional speech recognition models struggle with dysarthric speech due to:

* high articulation variability
* phoneme distortions and substitutions
* limited labeled clinical data

This project addresses these challenges by:

1. **Learning strong speech representations** using self-supervised learning (SSL)
2. **Explicitly modeling phoneme behavior** using symbolic constraints
3. **Explaining recognition errors** through phoneme-level and rule-based analysis

---

### System Overview (Conceptual)

**Input → Representation → Reasoning → Output → Explanation**

1. **Speech Input (Dysarthric & Control)**

   * Raw audio recordings from a clinical speech dataset
   * Includes speaker variability and dysarthria-specific patterns

2. **Self-Supervised Speech Encoder**

   * A pretrained SSL model extracts high-level acoustic–phonetic representations
   * Reduces dependence on large labeled datasets
   * Captures phoneme-like latent units crucial for impaired speech

3. **Phoneme-Level Modeling**

   * Neural representations are mapped to phoneme probabilities
   * Enables fine-grained analysis beyond word-level predictions

4. **Neuro-Symbolic Constraint Layer**

   * Incorporates symbolic phoneme rules such as:

     * common dysarthric substitutions
     * articulatory similarity constraints
   * Acts as a reasoning layer that guides or regularizes neural predictions

5. **ASR Decoding**

   * Produces word- or sentence-level transcription
   * Uses phoneme-aware outputs rather than purely acoustic decisions

6. **Explainability Module**

   * Provides:

     * phoneme attribution for errors
     * symbolic rule activations
     * interpretable error categories
   * Enables clinical and academic interpretability

---

### Key Characteristics

* **Neuro-Symbolic**: Combines neural perception with symbolic reasoning
* **Self-Supervised**: Leverages pretrained speech knowledge efficiently
* **Explainable**: Avoids black-box behavior common in deep ASR
* **Clinically Relevant**: Tailored to dysarthric speech characteristics

---

### Expected Outcomes

* Improved robustness on dysarthric speech compared to baseline ASR
* Clear phoneme-level explanation of recognition errors
* Insight into how symbolic constraints influence neural predictions
* A modular architecture suitable for research extensions

---
