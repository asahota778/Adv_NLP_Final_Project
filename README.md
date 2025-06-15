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

This project focuses on **multi-class patent classification** using the BigPatent dataset. The primary objective is to automatically classify patent abstracts into one of nine CPC (Cooperative Patent Classification) categories using only the abstract text, exploring various NLP techniques from rule-based baselines to state-of-the-art transformer models.

### Business Case
Patent offices worldwide face the challenge of manually classifying thousands of patent applications daily. Automating this process can:
- Reduce classification time from hours to seconds
- Improve consistency across patent examiners
- Enable better technology trend analysis
- Support intellectual property portfolio management

### Dataset Overview
- **Source**: [BigPatent Dataset](https://huggingface.co/datasets/NortheasternUniversity/big_patent)
- **Size**: 67,068 patent abstracts (after removing duplicates)
- **Classes**: 9 CPC categories with significant class imbalance
- **Split**: 60/20/20 (train/validation/test)

---

## Project Structure and Summaries

### Part 0: Dataset Selection
**Objective**: Choose a unique text classification dataset from HuggingFace

**Summary**: Selected the BigPatent dataset for its real-world applicability, technical complexity, and challenging class imbalance. The dataset represents actual U.S. patent documents classified into CPC categories.

### Part 1: Setting Up the Problem (1.5 points)
**Objective**: Establish baselines and understand dataset characteristics

**Key Results**:
- **Dataset Analysis**: Severe class imbalance with classes 'g' (Physics) and 'h' (Electricity) at ~21% each, while 'd' (Textiles) represents only 0.84%
- **Random Baseline**: Stratified random classifier achieves 15.74% accuracy
- **Rule-Based Classifier**: Domain keyword approach reaches 32.27% accuracy
- **TF-IDF + Logistic Regression**: Statistical baseline achieves **58.16% accuracy** and **52.12% macro F1**

**Key Finding**: The task is moderately challenging with clear technical distinctions between classes, but minority classes pose significant challenges.
