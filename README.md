# Advanced NLP Final Project


# Part 3 - NLP Pipeline for Patent Text Classification

## Overview

This project implements a comprehensive machine learning pipeline for classifying patent abstracts using advanced natural language processing techniques. The pipeline addresses the challenge of text classification in a domain-specific context where class imbalance is a significant concern. The approach combines state-of-the-art transformer models with sophisticated data preprocessing, hyperparameter optimization, and evaluation strategies to achieve robust performance across multiple experimental settings.

## Data Preprocessing and Dataset Preparation

### Initial Dataset Characteristics

The dataset consists of patent abstracts with severe class imbalance, containing 9 different classes labeled from 0 to 8. The original distribution shows significant variation across classes, with some classes having as few as 561 samples while others contain over 14,000 samples. This imbalance presents a significant challenge for machine learning models, as they tend to bias toward majority classes and perform poorly on minority classes without proper handling.

### Strategic Dataset Splitting Approach

The data splitting strategy is carefully designed to ensure fair evaluation while maintaining the integrity of the original data distribution in test scenarios. The process begins with a stratified split of the complete dataset into training plus validation (80%) and test (20%) portions. This initial split preserves the class proportions across both portions, ensuring that the test set remains representative of the original data distribution.

The training plus validation portion is then further divided using another stratified split, allocating 75% to training and 25% to validation. This results in a final distribution where 60% of the original data becomes the training set, 20% becomes the validation set, and 20% remains as the test set. The stratified approach ensures that each split maintains the original class proportions, which is crucial for reliable model evaluation.

### Targeted Class Balancing Strategy

The class balancing approach represents a sophisticated solution to the imbalance problem. Rather than applying balancing techniques to the entire dataset, which would artificially alter the natural distribution in evaluation sets, the balancing is applied exclusively to the training set. This approach allows the model to learn from balanced data while still being evaluated on realistic, naturally distributed data.

The balancing process sets a target of 5,000 samples per class in the training set. For classes with fewer than 5,000 samples, oversampling with replacement is employed, where existing samples are randomly duplicated until the target is reached. For classes exceeding 5,000 samples, undersampling without replacement is used, randomly selecting 5,000 samples from the available data. This results in a perfectly balanced training set containing 45,000 samples total, with exactly 5,000 samples representing each of the 9 classes.

### Evaluation Set Integrity

The validation and test sets deliberately maintain their natural class distributions. This design choice is critical for realistic performance assessment, as real-world deployment scenarios will encounter the same imbalanced distributions present in the original data. The validation set contains 13,148 samples with class distributions ranging from 112 to 2,821 samples per class, while the test set contains an identical number of samples with similar natural distribution patterns.

## Hyperparameter Optimization Framework

### Optuna Integration and Configuration

The hyperparameter optimization employs Optuna, a state-of-the-art optimization framework that provides intelligent parameter exploration capabilities. The configuration is set to run 10 trials with a maximum timeout of one hour, balancing computational efficiency with thorough parameter exploration. The framework uses a Tree-structured Parzen Estimator (TPE) sampler, which learns from previous trials to make informed suggestions for subsequent parameter combinations.

The optimization process also incorporates a Median Pruner for early stopping of unpromising trials. This pruner monitors intermediate results and terminates trials that perform significantly worse than the median performance of previous trials at the same stage. This approach significantly improves computational efficiency by avoiding the completion of clearly suboptimal parameter combinations.

### Parameter Space and Model Architecture

The optimization focuses on three critical hyperparameters that significantly impact model performance. The number of training epochs is varied between 1 and 5, allowing the system to find the optimal balance between training thoroughness and overfitting prevention. The batch size is selected from options of 4, 8, 16, and 32, which affects both memory usage and gradient estimation quality. The maximum sequence length is chosen from 128, 256, and 512 tokens, balancing computational efficiency with the ability to capture longer text sequences.

The underlying model architecture uses SetFit with the AI-Growth-Lab/PatentSBERTa backbone, which is specifically trained on patent documents. This specialized model provides superior performance on technical and scientific text compared to general-purpose language models. The SetFit framework combines the efficiency of sentence transformers with the effectiveness of few-shot learning approaches, making it particularly suitable for domain-specific classification tasks.

### Optimization Execution and Model Selection

During the optimization process, each trial creates a fresh model instance with the suggested hyperparameters and trains it on a subset of the training data. The model is then evaluated on a portion of the validation set using macro F1 score as the primary optimization metric. Macro F1 score is chosen because it provides equal weight to all classes regardless of their frequency, making it particularly appropriate for imbalanced datasets.

After all trials complete, the framework identifies the hyperparameter combination that achieved the highest validation performance. These optimal parameters are then used to train a final model on the complete training dataset, providing the best possible performance for subsequent experiments. The entire optimization process includes comprehensive error handling and resource management to ensure stable execution even with GPU memory constraints.

## Learning Curve Analysis and Performance Scaling

### Experimental Design and Methodology

The learning curve analysis provides crucial insights into how model performance scales with the amount of training data available. This analysis uses the optimized hyperparameters discovered through the Optuna process, ensuring that each data point represents the best possible performance achievable with that amount of training data. The experiment systematically evaluates model performance using 1%, 10%, 25%, 50%, 75%, and 100% of the balanced training set.

For each data percentage, a stratified sampling approach creates a subset that maintains the balanced class distribution established during preprocessing. This ensures that even small subsets contain representative samples from all classes, preventing the analysis from being confounded by missing classes in smaller training sets. Each subset is used to train a fresh model instance with the optimal hyperparameters, providing clean measurements of performance scaling.

### Training and Evaluation Protocol

The training protocol for each data percentage follows a standardized approach to ensure comparable results. Each model is initialized with the same pretrained PatentSBERTa weights and trained using the optimized hyperparameters determined through the Optuna process. The training process includes comprehensive monitoring of both training and validation performance to detect potential overfitting and assess generalization capability.

Evaluation is performed on a consistent validation subset throughout all experiments, ensuring that performance comparisons are fair and meaningful. The validation set maintains its natural class distribution, providing realistic assessment of how models will perform in deployment scenarios. Multiple metrics are tracked including accuracy, macro F1 score, micro F1 score, and training time, providing a comprehensive view of model performance characteristics.

### Performance Metrics and Analysis

The learning curve analysis tracks both training and validation performance to provide insights into model behavior across different data scales. Training performance indicates how well the model can fit the available data, while validation performance reveals generalization capability. The gap between these metrics helps identify optimal training set sizes and detect potential overfitting issues.

Macro F1 score serves as the primary performance metric due to its appropriateness for imbalanced datasets, giving equal weight to performance on all classes regardless of their frequency. Micro F1 score is also tracked to provide overall prediction accuracy, while training time measurements help assess computational efficiency scaling. These comprehensive metrics enable identification of the optimal balance between performance, data requirements, and computational cost.

## Test Set Evaluation and Generalization Assessment

### Comprehensive Model Evaluation Strategy

The test set evaluation phase provides the final assessment of model performance across all training data scales. After completing the learning curve analysis, every successfully trained model is evaluated on the held-out test set to measure real-world performance and generalization capability. This evaluation uses the same test set that was created during the initial data splitting process and has remained completely isolated from all training and hyperparameter optimization procedures.

The test set contains 13,148 samples representing 20% of the original dataset, with the natural class distribution preserved. This means that the test evaluation reflects realistic deployment scenarios where models encounter the same imbalanced class distributions present in real-world data. The evaluation process systematically applies each trained model to predict labels for the entire test set, calculating comprehensive performance metrics for comparison and analysis.

### Performance Measurement and Analysis

The test evaluation calculates multiple performance metrics to provide a complete picture of model capabilities. Test accuracy measures overall prediction correctness, while macro F1 score provides class-balanced performance assessment that is crucial for imbalanced datasets. Micro F1 score offers an alternative perspective on overall performance, and inference time measurements help assess practical deployment considerations.

A critical component of the test evaluation is the analysis of generalization gaps between validation and test performance. Large gaps may indicate overfitting to the validation set, while small gaps suggest good generalization capability. This analysis helps identify the most reliable models and optimal training data sizes for practical deployment. The evaluation also tracks how test performance scales with training data size, providing insights into data efficiency and the point of diminishing returns for additional training data.

### Comparative Analysis and Model Selection

The test evaluation includes comprehensive comparative analysis across all training data percentages to identify optimal strategies. Performance scaling analysis reveals how test accuracy and F1 scores improve with additional training data, helping determine the most efficient use of labeling resources. The evaluation identifies both the best performing models in absolute terms and the most data-efficient approaches that achieve strong performance with minimal training data.

The analysis also examines the consistency of model performance across different metrics and identifies any trade-offs between accuracy and F1 scores. This information is crucial for practical deployment decisions, where the choice between maximizing overall accuracy versus ensuring balanced performance across all classes depends on specific application requirements.

## Technique Comparison and Methodological Analysis

### Experimental Framework for Technique Evaluation

The technique comparison provides a systematic evaluation of three distinct machine learning approaches applied to the same patent classification problem. This comparison is designed to assess the relative effectiveness of different strategies when labeled data is limited, which is a common scenario in domain-specific applications like patent analysis. Each technique represents a different philosophy for handling limited labeled data, providing insights into the trade-offs between data requirements, computational cost, and performance.

The comparison uses a carefully controlled experimental design where all techniques are evaluated on the same validation set and use the same underlying model architecture. This ensures that performance differences reflect the effectiveness of the techniques themselves rather than variations in evaluation criteria or model capabilities. The evaluation includes both performance metrics and practical considerations such as training time and data labeling requirements.

### Few-Shot Learning Approach

The few-shot learning technique represents the most data-efficient approach in the comparison, using only 32 carefully selected training samples distributed across all 9 classes. The sample selection process ensures balanced representation with approximately 3-4 samples per class, though some classes receive one additional sample to reach the target of 32 total samples. This minimal training set is designed to test the limits of what can be achieved with extremely limited labeled data.

The few-shot model uses extended training with 4 epochs to maximize learning from the limited available data. This approach leverages the pretrained PatentSBERTa model's existing knowledge of patent language and technical terminology, fine-tuning it with the small labeled dataset. The technique is particularly relevant for scenarios where obtaining labeled data is expensive or time-consuming, making it important to understand the baseline performance achievable with minimal annotation effort.

### Pseudo-Labeling Semi-Supervised Strategy

The pseudo-labeling approach implements a sophisticated semi-supervised learning strategy that begins with the few-shot model and iteratively expands the training set with high-confidence predictions. This technique starts by training the few-shot model on the 32 labeled samples, then uses this model to generate predictions for a larger pool of unlabeled data. The approach carefully filters these predictions, retaining only those with confidence scores above 0.8 to ensure high-quality pseudo-labels.

The pseudo-labeling process selects the top 5 highest-confidence predictions for each class, adding these pseudo-labeled samples to the original training set. This expansion maintains class balance while significantly increasing the effective training data size. The model is then retrained on the combined dataset of original labeled samples plus pseudo-labeled samples, potentially achieving better performance than either fully supervised learning with equivalent labeled data or the initial few-shot approach.

### Full Supervised Learning Baseline

The full supervised learning approach serves as a baseline for comparison, using an amount of fully labeled data equivalent to the combined dataset used in pseudo-labeling. This technique represents traditional supervised learning without any special handling for limited data scenarios. The approach trains directly on labeled data using standard procedures, providing a reference point for assessing whether the more sophisticated semi-supervised approaches offer genuine advantages.

The supervised learning baseline uses the same model architecture and training procedures as the other techniques, differing only in the training data composition. This approach helps quantify the value of pseudo-labeling by comparing it against simply using more labeled data. The comparison reveals whether the effort of implementing semi-supervised learning provides benefits over straightforward data labeling efforts.

### Comparative Evaluation and Practical Implications

The technique comparison evaluates each approach across multiple dimensions including prediction accuracy, macro F1 score for balanced class assessment, training time, and data labeling requirements. The analysis considers both absolute performance and data efficiency, calculating performance per labeled sample to assess the value of different approaches when labeling budget is constrained.

The comparison provides practical guidance for selecting appropriate techniques based on specific constraints and requirements. For scenarios with extremely limited labeling budgets, the few-shot learning results indicate baseline performance expectations. The pseudo-labeling evaluation reveals whether semi-supervised approaches can bridge the gap between few-shot and fully supervised performance. The full supervised baseline quantifies the performance ceiling achievable with adequate labeled data, helping inform decisions about labeling investment versus technique sophistication.

## Technical Implementation Details

### Model Architecture and Configuration

The implementation centers around the SetFit framework combined with the specialized PatentSBERTa language model, creating a powerful combination for patent text classification. PatentSBERTa is specifically pretrained on patent documents, giving it superior understanding of technical terminology, patent language patterns, and domain-specific concepts compared to general-purpose language models. The SetFit framework provides an efficient approach to fine-tuning sentence transformers for classification tasks, offering better performance than traditional approaches while requiring less computational resources.

The model configuration includes careful attention to hardware optimization, with automatic GPU detection and memory management to handle the computational demands of transformer models. The implementation includes comprehensive error handling and resource cleanup to ensure stable execution across different hardware configurations and prevent memory leaks during extended training sessions.

### Data Processing and Quality Control

The data processing pipeline implements robust preprocessing steps to ensure data quality and consistency throughout the experimental process. Text normalization handles various formatting issues commonly found in patent abstracts, while label encoding converts categorical labels to numerical format required by machine learning models. The preprocessing includes validation steps to detect and handle missing values, duplicate entries, and potential data corruption issues.

The implementation carefully tracks data lineage throughout all processing steps, ensuring that train, validation, and test sets remain properly isolated and that any sampling or balancing operations are reproducible. Random seed management ensures that all experimental results can be replicated, which is crucial for scientific validity and debugging purposes.

### Evaluation Framework and Metrics

The evaluation framework implements comprehensive assessment capabilities with multiple metrics calculated consistently across all experiments. The framework handles the complexities of multi-class classification with imbalanced data, providing both overall performance measures and class-specific assessments. Metric calculations include proper handling of edge cases such as classes with zero predictions and appropriate averaging methods for multi-class scenarios.

The implementation includes extensive visualization capabilities for learning curves, performance comparisons, and technique evaluations. These visualizations provide intuitive understanding of model behavior and performance trends, supporting both technical analysis and communication of results to stakeholders with varying technical backgrounds.

## Conclusion and Key Insights

This comprehensive machine learning pipeline demonstrates sophisticated approaches to handling domain-specific text classification challenges, particularly in scenarios involving class imbalance and limited labeled data. The systematic experimental design provides valuable insights into the trade-offs between different techniques and the scaling behavior of modern transformer-based models.

The pipeline's emphasis on maintaining realistic evaluation conditions while applying advanced balancing techniques to training data represents a mature approach to machine learning system design. The combination of automated hyperparameter optimization, comprehensive learning curve analysis, and systematic technique comparison provides a robust framework for developing and evaluating text classification systems in challenging real-world scenarios.

