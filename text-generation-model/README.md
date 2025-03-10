# Project Title: PblockerAI

## Overview

This project implements a text generation AI model using state-of-the-art techniques in natural language processing. The model is designed to generate coherent and contextually relevant text based on the input it receives.

## Project Structure

The project is organized as follows:

```
text-generation-model
├── configs
│   ├── model_config.json       # Configuration settings for the model architecture
│   └── training_args.json      # Training parameters for the model
├── data
│   ├── test.csv                # Evaluation samples for testing the model
│   └── train.csv               # Training samples for training the model
├── src
│   ├── train_chatbot.py        # Main script for training the text generation model
│   └── utils.py                # Utility functions for data preprocessing and evaluation
├── LICENSE                      # Licensing information for the project
├── requirements.txt            # List of Python dependencies required for the project
└── README.md                   # Documentation for the project
```

# Model change
- Change ```gpt2``` to ```gpt2-medium``` for pb-large-2.5

## Setup Instructions

1. Prepare your datasets in the `data` directory. Ensure that `train.csv` contains the training samples and `test.csv` contains the evaluation samples.

2. Configure the training parameters in `configs/training_args.json` and the model architecture in `configs/model_config.json`.

3. Run the training script:
   ```
   python src/train_chatbot.py
   ```

## Model Details

The model is designed for text generation tasks and is built using state-of-the-art techniques in natural language processing. The architecture and training configurations can be adjusted through the respective configuration files.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.