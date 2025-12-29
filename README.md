# Detecting Early Signs of Dementia from Spontaneous Speech using a Hybrid Wav2Vec2 and SVM Architecture

## 1. Problem Framing

The Challenge Dementia, particularly Alzheimer's Disease (AD), is a growing global health crisis. According to the World Health Organization, over 55 million people live with dementia worldwide. The traditional path to diagnosis is often invasive (lumbar punctures), expensive (PET scans), or requires lengthy cognitive tests administered by specialists. Consequently, diagnosis often comes too late for effective intervention or lifestyle management.

The Opportunity: Speech as a Biomarker Speech production involves complex cognitive planning. Even in early stages, neurodegenerative diseases manifest subtle acoustic and linguistic anomalies—such as hesitations, changes in speech rate, and intonation flattening—that are imperceptible to the human ear but detectable by AI.

The Core Problem: Data Scarcity in Medical AI While Deep Learning models like Transformers have revolutionized audio analysis, they typically require massive datasets to learn effectively. In the medical domain, high-quality labeled data is scarce. In our initial experiments, we found that standard fine-tuning of large Transformer models (like Wav2Vec2) on small datasets (<300 samples) leads to severe instability and overfitting, resulting in poor generalization (accuracy hovering around ~53%, equivalent to random guessing).

Project Objective Our goal was to build a robust, non-invasive screening tool capable of detecting dementia from short audio clips of spontaneous speech. We aimed to overcome the "small data" limitation by engineering a Hybrid Architecture that combines the feature extraction power of Self-Supervised Learning (SSL) with the stability of classical Machine Learning.
