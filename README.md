
# BERT-Based Text Classification with Evaluation Metrics

## Introduction
This notebook demonstrates the process of building, training, and evaluating a text classification model using the BERT (Bidirectional Encoder Representations from Transformers) architecture. The model is designed to classify textual data into multiple categories using the Hugging Face `transformers` library. The notebook covers the following key steps:

1. **Data Preparation**: Encoding text data and labels for training.
2. **Model Setup**: Initializing and configuring the BERT model for sequence classification.
3. **Training Process**: Implementing a training loop with the AdamW optimizer and learning rate scheduler.
4. **Evaluation**: Assessing model performance on the validation set using metrics like accuracy and F1 score.

## Data Preparation

### a. Import Libraries and Modules
The first step involves importing necessary libraries including PyTorch, Hugging Face `transformers`, and scikit-learn. These libraries provide tools for model building, training, and evaluation.

### b. Load and Encode Data
The data (training and validation) is loaded into DataFrames. The labels are encoded using `LabelEncoder` from scikit-learn to convert categorical labels into integers that can be processed by the model.

#### Sample Data
```
                                                text label
0    We extend to natural deduction the approach ...    LO
1    Over the last decade, the IEEE 802.11 has em...    NI
2    Motivated by the problem of storing coloured...    DS
3    We consider the downlink of a cellular syste...    NI
4    Meroitic is the still undeciphered language ...    CL
```

### c. Create Custom Dataset Class
A custom `Dataset` class is defined to handle the text and label data. This class preprocesses the text (tokenization, attention masks) and prepares it for the model.

### d. DataLoader Setup
`DataLoader` objects are created to handle batching and shuffling during training and validation. This ensures efficient loading of data in batches.

## Model Setup

### a. Load BERT Model and Tokenizer
The BERT tokenizer and sequence classification model are loaded from the Hugging Face `transformers` library. The model is set up to classify the input text into 7 different categories.

### b. Move Model to GPU
The model is moved to the GPU to leverage hardware acceleration during training, which significantly speeds up the process.

## Training Process

### a. Initialize Optimizer and Scheduler
The AdamW optimizer is used to update the model parameters, and a learning rate scheduler is configured to adjust the learning rate over time.

### b. Training Loop
The training loop iterates over the dataset for a specified number of epochs, updating the model parameters and calculating the loss for each batch. The learning rate is adjusted using the scheduler after each batch.

#### Training Output
```
Epoch 1, Loss: 0.4883, Accuracy: 0.8489
Epoch 2, Loss: 0.2260, Accuracy: 0.9289
Epoch 3, Loss: 0.1305, Accuracy: 0.9627
```

## Model Evaluation

### a. Model Evaluation Mode
The model is set to evaluation mode to disable dropout and other training-specific behaviors.

### b. Predict on Validation Set
Predictions are made on the validation set without calculating gradients. These predictions are stored for later comparison with the true labels.

### c. Convert Predictions to CPU and Evaluate
After obtaining predictions on the GPU, they are moved to the CPU and converted to NumPy arrays to be processed by scikit-learn for generating a classification report.

#### Sample Evaluation Output
```
Weighted F1 Score on the validation set: 0.9127
```

## Conclusion
This notebook provides a comprehensive pipeline for text classification using BERT. It starts with data preparation, moves through model training, and concludes with evaluation on the validation set. The use of GPU accelerates the process, and detailed evaluation metrics, including accuracy and F1 score, are provided to assess model performance.
