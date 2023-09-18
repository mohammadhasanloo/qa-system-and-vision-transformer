# NN-CA5-1.QA-System-2.CIFAR-10-Image-Classification:-Leveraging-Vision-Transformers

### 1.QA System [Link](#part-1-qa-system)

### 2.CIFAR-10 Image Classification: Leveraging Vision Transformers [Link](#part-2-cifar-10-image-classification-leveraging-vision-transformers)

# Part 1: QA System

## Problem Modeling

BERT is built upon the Transformer architecture, consisting of a stack of encoder layers. Each encoder layer has a self-attention mechanism, multi-head attention, and a feed-forward neural network. This architecture allows BERT to understand bidirectional word relationships. BERT utilizes a two-step process (fine-tuning and pre-training). In the pre-training phase, the model is trained on a large corpus of unlabeled text using two unsupervised tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). In the subsequent phase, BERT is fine-tuned for specific tasks by adding task-specific layers and using labeled data.

For example, in the MLM task, sequences are masked (some of their elements are hidden) and fed as input to the model, which attempts to reconstruct the original sequence in the output. In the NSP task, a sequence is provided as input, and the model tries to predict the next sentence. Entropy cross-entropy loss can be used as the loss function for these tasks. The final layers of these tasks typically map the feature space to token space, and thus, removing these layers after pre-training results in a highly efficient encoder.

For the extractive question answering task, our input consists of a question sequence concatenated with a context sequence. To achieve this, we use a Transformer model tokenizer. Typically, the tokenizer automatically uses the [SEP] token to separate the two sequences. The model's output should include two vectors of the same sequence length: one indicating the start position of the answer in the input sequence and the other indicating the end position.

As a result, we can add two separate neurons at the end of the encoders to assign a probability to each token in the feature space, indicating how likely it is for the current token to be the start or end of the answer to the question in the input sequence. We can then use the cross-entropy loss function for training the model.

## Data Preprocessing

Contrary to what the paper states, there were no questions with multiple answers in the cloned dataset from GitHub. However, we load the data, initially in JSON format, and convert them into a dataset consisting of point data, each containing a question, context, and answer. This conversion allows for easier preprocessing and training.

Statistical information about the dataset for each section is observed in the following sections.

Next, we preprocess the data. This section's explanations pertain to post-processing since we preprocess and standardize all training, evaluation, and test data in this section.

First, we tokenize the data according to the initial descriptions. As seen in the statistics for various dataset sections, approximately 25% of questions have no answers, and there are no answers in the context. Additionally, in some data, the total length of the question and context exceeds the maximum allowed input sequence length, which is 128 in this case. To address this issue, we initially set the output for all questions that cannot be answered based on their context to the position of the [CLS] token. Additionally, if the question and context sequence exceeds the model's maximum input sequence length, we split the context sequence into multiple overlapping segments of equal length and consider each of these segments, along with the corresponding question, as separate data points. To determine the output, we check each context segment to see if a complete answer exists. If it does, we consider the start and end positions of that answer as labels. If a complete answer does not exist in that segment, we treat the data in that segment as impossible to answer, as per the previous agreement.

If the question and context sequence is shorter than the model's input, we pad the sequence with special [PAD] tokens. This way, we tokenize all data and format it properly for input into the model.

## Model Implementation

We load base models using TFAutoModel in the Tensorflow environment. Since model training is time-consuming, with each epoch taking approximately one hour, code for both ParsBERT and ALBERT has been implemented in separate files. However, all code sections in these two files are identical, with the only difference being the base model used in constructing the QA model.

We use a custom loss function that calculates the cross-entropy loss separately for two model outputs. The final loss of the model is the sum of these two losses.

In this task, since we use pre-trained models, our goal is not to train the model from scratch but to perform fine-tuning. Therefore, we set a low learning rate, around 10^-4. According to our experiments, values in the range of 10^-5 yield good results.

Additionally, due to ParsBERT being larger than ALBERT, the convergence speed of ALBERT is higher than ParsBERT. Consequently, we trained the ParsBERT model for two epochs and the ALBERT model for one epoch.

## Evaluation and Post-processing

After completing the training of the models, we evaluate the performance of the models using the mentioned metrics and the test dataset. The table below compares the model results with the results from the paper.

Table 1 – Model Results Compared to Paper Results

| Model    | Exact Match | F1 Score | EM - Paper | F1 - Paper |
| -------- | ----------- | -------- | ---------- | ---------- |
| ParsBERT | 67.8%       | 74.4%    | 68.1%      | 82.0%      |
| ALBERT   | 69.2%       | 76.2%    | -          | -          |

In the following sections, we also observe examples of the model's responses to test data, where both models provide the same answer.

# Part 2: CIFAR-10 Image Classification: Leveraging Vision Transformers

## Introduction

This repository contains code for a comparative study of two different architectures for image classification: Convolutional Neural Networks (CNN) and Vision Transformers (ViT). We implemented the code based on the techniques described in the research paper titled ["Training data-efficient image transformers & distillation through attention"](https://arxiv.org/abs/2012.12877). In this README, we will explain the preprocessing steps, model architectures, training settings, and present the results of our experiments.

## Data Preprocessing

We begin by performing essential data preprocessing using PyTorch's transforms:

- Scaling: Images are scaled up from (3, 32, 32) to (3, 224, 224).
- Data Augmentation: Random horizontal flipping is applied to augment the dataset.
- Tensor Conversion: Data is converted into tensors.
- Normalization: Data is normalized.

These preprocessing steps are illustrated in Figure 8 in the paper.

## Dataset Loading

We load the 10CIFAR dataset using the torchvision library. The training data is shuffled to ensure no bias is introduced by the image order.

## Batch Size

Initially, we set the batch size to 512 as recommended by the paper. However, during model training, we encountered memory fragmentation errors. After experimentation, we reduced the batch size to 32, which resolved the issue. We adopted this batch size for both models for fair comparison.

## Convolutional Neural Network (CNN)

In this study, we implemented several convolutional neural network architectures. One of them is the 19VGG model, in which one trainable convolutional layer from the fifth block is unfrozen. The 29th layer is unfrozen to enable better classification.

### Model Architecture

We define a sequential model for the CNN:

1. Flattening: The output of the convolutional network is flattened.
2. Dense Layers: The flattened output is connected to 256 neurons using the ELU activation function. A dropout layer with a 50% probability is applied to reduce overfitting.
3. Classification: The model is connected to 10 neurons (matching the 10CIFAR classes), and a softmax layer is implemented in the fn_accuracy function to provide class probabilities.

### Loss Function and Learning Rate Scheduler

We calculate the loss using cross-entropy, suitable for classification tasks, and employ a learning rate scheduler, following the paper's recommendations.

### Results

After fine-tuning, the convolutional model achieved an impressive 99.9% accuracy on the training data and 91.2% on the evaluation data. This process took 20 epochs, equivalent to 7,294 seconds or approximately 2 hours.

## Vision Transformer (ViT)

For the Vision Transformer (ViT), we use Deit as specified in the paper. The 12th block is unfrozen.

### Model Architecture

Similar to the CNN architecture, we define a sequential model for ViT:

1. Flattening: The output of the transformer network is flattened.
2. Dense Layers: The flattened output is connected to 256 neurons using the ELU activation function. A dropout layer with a 50% probability is applied to reduce overfitting.
3. Classification: The model is connected to 10 neurons (matching the 10CIFAR classes), and a softmax layer is implemented in the fn_accuracy function to provide class probabilities.

### Loss Function and Learning Rate Scheduler

Loss calculation and learning rate scheduling are consistent with the paper's recommendations.

### Results

Considering that each epoch took approximately half an hour, we trained the ViT model for only 5 epochs, totaling 8,501 seconds or roughly 2 hours and 20 minutes.

## Comparison of Results

The table below compares the results obtained from the CNN and ViT models with the results reported in the paper:

| Model Type                  | Consumed Time (seconds) | # of Epochs in Project | Validation Accuracy (%) in Project | # of Epochs in Paper | Validation Accuracy (%) in Paper |
| --------------------------- | ----------------------- | ---------------------- | ---------------------------------- | -------------------- | -------------------------------- |
| VGG-19 (With Unfreezing)    | 92.784                  | 20                     | 91.2                               | 20                   | 72.90                            |
| VGG-19 (Without Unfreezing) | -                       | -                      | 87.1                               | 20                   | 72.90                            |
| Deit Transformer            | 96.450                  | 20                     | 96.1                               | 5                    | 96.10                            |

As evident, the ViT model outperforms the CNN model, albeit with fewer epochs. Deit achieved a 0.35% improvement in accuracy compared to the paper's results. Despite the longer time required for ViT due to its complex architecture and a larger number of parameters, the significant performance gain is notable.

Moreover, it's worth mentioning that unfreezing the 29th layer of the VGG-19 model resulted in a significant 4% difference compared to freezing it. In comparison to the paper's results, VGG-19 showed a 1.6% difference under equivalent conditions. Deit achieved a 0.35% difference with 15 fewer epochs.
