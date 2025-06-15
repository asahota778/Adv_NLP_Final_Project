# Advanced Methods in Natural Language Processing - Final Project

**Executive Summary**

Submitted by:
- Hannes Schiemann (<hannes.schiemann@bse.eu>)
- Angad Singh Sahota (<angad.sahota@bse.eu>)
- Deepak Malik (<deepak.malik@bse.eu>)
- Nicolas Rauth (<nicolas.rauth@bse.eu>)

*Barcelona School of Economics*  
**MSc in Data Science for Decision Making**

Submitted to: **Arnault Gombert** (<arnault.gombert@bse.eu>)

---

## Main Task and Objective

This project focuses on **multi-class patent classification** using a subsample of the BigPatent dataset (https://huggingface.co/datasets/NortheasternUniversity/big_patent) consisting of around 1.2 Million patent abstracts. We are just using the validation set consisting of around 67 000 abstracts. The primary objective is to automatically classify patent abstracts into one of nine CPC (Cooperative Patent Classification) categories using only the abstract text, exploring various NLP techniques from rule-based baselines to state-of-the-art transformer models.

### Business Case
Patent offices worldwide face the challenge of manually classifying thousands of patent applications daily. Automating this process can:
- Reduce classification time from hours to seconds
- Improve consistency across patent examiners
- Enable better technology trend analysis
- Support intellectual property portfolio management

### Dataset Overview
- **Source**: [BigPatent Dataset](https://huggingface.co/datasets/NortheasternUniversity/big_patent)
- **Size**: 67,068 patent abstracts 
- **Classes**: 9 CPC categories with significant class imbalance
- **Split**: 60/20/20 (train/validation/test)

---

## Project Structure and Summaries

### Part 0: Dataset Selection
**Objective**: Choose a unique text classification dataset from HuggingFace

**Summary**: Selected the BigPatent dataset for its real-world applicability, technical complexity, and challenging class imbalance. The dataset represents actual U.S. patent documents classified into CPC categories.

### Part 1: Setting Up the Problem (1.5 points)
**Objective**: Establish baselines and understand dataset characteristics

- **Dataset Analysis**: Severe class imbalance with classes 'g' (Physics) and 'h' (Electricity) at ~21% each, while 'd' (Textiles) represents only 0.84%
- **Random Baseline**: Stratified random classifier achieves 15.74% accuracy
- **Rule-Based Classifier**: Domain keyword approach reaches 32.27% accuracy
- **TF-IDF + Logistic Regression**: Statistical baseline achieves **58.16% accuracy** and **52.12% macro F1**

**Key Finding**: The task is moderately challenging with clear technical distinctions between classes, but minority classes pose significant challenges.

### Part 2: Data Scientist Challenge
**Objective**: Explore different techniques to enhance model performance with limited labeled data

**Key Results**: 
We trained a SetFit Model on 32 labeled abstracts achieving a macro-f1 score of aorund 0.3.
We the explored generating pseudo-labeled data and generating synthetic data with an LLM which improved performance by a bit.
The SetFit model was outperformed by Zero-Shot Classification using an LLM achieving a f1-score of 0.42.


### Part 3: Learning Curves and Technique Comparison

This part of the project focuses on analyzing the model's learning behavior and comparing different training strategies to understand their effectiveness, especially in data-scarce scenarios.

#### Learning Curve Analysis

* **Objective**: To understand how the model's performance improves as the amount of training data increases.
* **Summary**: Using the best hyperparameters found previously, a learning curve analysis is conducted. The model is trained on progressively larger subsets of the data (from 1% to 100%). The results are plotted to visualize how accuracy and F1-score scale with training set size, helping to diagnose whether the model suffers from high bias (underfitting) or high variance (overfitting).

#### Technique Comparison

* **Objective**: To compare the effectiveness of different training strategies, especially in low-resource scenarios.
* **Summary**: This experiment compares three distinct training techniques:
   1. **Few-shot Learning**: Training a model on a minimal, balanced dataset of just 32 samples.
   2. **Pseudo-labeling**: Using the few-shot model to generate labels for a larger, unlabeled dataset and then retraining on the combined data.
   3. **Standard Supervised Training**: Training a model on a larger, fully labeled dataset to serve as a baseline for comparison.

### Part 4: Model Optimization and Performance Comparison

This part focuses on creating a more efficient version of the best-performing model and conducting a detailed comparison of its performance, speed, and size against the original.

#### Model Distillation & Quantization

* **Objective**: To create a smaller, faster version of the best model while minimizing the impact on performance.
* **Summary**: The best-performing `SetFit` model is designated as the "teacher" to train a smaller `DistilBERT` "student" model through **knowledge distillation**. This process transfers the teacher's learned knowledge to the more compact student architecture. The student model is then further compressed using **quantization**, which reduces its memory footprint and can accelerate inference.

#### Performance and Speed Comparison

* **Objective**: To quantify the trade-offs between the original model and the optimized student model.
* **Key Results**:
   * The distilled student model is **2.0x faster** during inference and **39% smaller** in size (255.4 MB vs. the teacher's 417.7 MB).
   * This efficiency gain comes at a cost to performance, with the student's accuracy dropping by 10 percentage points from 61.0% to 51.0%.
* **Key Finding**: Knowledge distillation successfully produces a significantly more efficient model suitable for production environments, but it highlights the classic trade-off between computational cost and predictive accuracy.
