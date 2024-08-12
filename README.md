# ImageCaptioningDL
Computer Vision Project - Enhancing Image Captioning with Advanced Deep Learning Techniques. 
By Anna Ivanchenko, Kimia Arfaie and Hamza Zafar

## Introduction

Caption generation is a complex and fascinating problem in the realm of artificial intelligence, requiring a blend of techniques from both computer vision and natural language processing (NLP). The task involves generating a coherent textual description for a given photograph, a process that not only demands accurate interpretation of visual content but also the ability to express these observations in naturalistic language.

### Motivation

The ability to automatically generate textual descriptions from images has significant implications across various fields, from aiding visually impaired individuals to enhancing the accessibility of digital content on the web. As digital images become increasingly prevalent, the need for effective and efficient automated captioning systems becomes more critical. This project is inspired by Andrej Karpathy's seminal work, specifically his blog post "The Unreasonable Effectiveness of Recurrent Neural Networks," and builds on concepts from the influential paper "Deep Visual-Semantic Alignments for Generating Image Descriptions.".

### Objectives

This project aims to:
1. Explore the intersection of computer vision and NLP to tackle the problem of image captioning.
2. Implement and evaluate a model that leverages both Convolutional Neural Networks (CNNs) for image feature extraction and Long Short-Term Memory (LSTM) networks to generate captions.
3. Introduce an attention mechanism to improve the relevance and accuracy of the generated captions by focusing on specific regions of the image during the captioning process.

### Methodology Overview

The approach taken in this project involves:
- Utilizing the Flickr8k dataset, which comprises approximately 8,091 images, each annotated with five different captions, providing a rich dataset for training and evaluating our models.
- Preprocessing images using the VGG16 architecture, adapted to extract meaningful feature vectors that capture the essence of the visual content.
- Developing an LSTM-based sequence model to construct captions from these features, incorporating an attention mechanism to enhance the model's focus on relevant image parts dynamically as it generates text.

## Dataset

For this project, the Flickr8k Dataset is utilized. This dataset comprises 8,091 images, each accompanied by five unique captions, totaling 40,455 captions.

Dataset Source: [Kaggle - Flickr8k](https://www.kaggle.com/shadabhussain/flickr8k)

## Model Architecture for Caption Generation

Our model is designed to integrate textual and visual information in a unified framework that facilitates effective image captioning. This section outlines the construction of the model, detailing each component's role:

### Components of the Model:
- **Image Features Input**: A dense layer that takes flattened image feature vectors (from VGG16) and projects them into a new space to create a more useful representation for captioning.
- **Text Input and Embedding**: Captions are input as sequences of integers, which are then embedded into a higher-dimensional space to capture semantic meanings of words effectively.
- **Combining Features**: The LSTM processes the text embeddings to capture the context within captions, while the image features provide visual context. These are combined using an additive operation to merge textual and visual cues.
- **Output Layer**: A dense layer with softmax activation computes the probability distribution over all words in the vocabulary for each position in the caption, facilitating the generation of the next word based on the current context.

### Importance of This Architecture:
- **Dual Input Paths**: By separately processing image and text inputs before combining them, our model can learn to balance the influence of visual and textual information effectively.
- **Sequential Processing of Text**: The LSTM's ability to handle sequences makes it ideal for caption generation, where the order of words and the context from previous text are crucial for coherent outputs.

### Model Summary Insights:
The summary indicates a complex model with several million trainable parameters, emphasizing the depth and capacity required to handle the nuances of image captioning. This architecture is designed to optimize performance by carefully tuning and balancing the contributions from both visual and textual inputs.
