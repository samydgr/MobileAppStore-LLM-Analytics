# MobileAppStore-LLM-Analytics

## Overview

**MobileAppStore-LLM-Analytics** is a powerful tool for extracting insights and performing analytics on mobile app store data using Large Language Models (LLMs). The project fine-tunes **BERT** for classifying whether an app is successful or not and leverages **Gemma-2-9B**, a more powerful LLM, to explain the reasons for failure.

## Features

- 📊 **Data Filtering & Exploration**: Users can filter and explore mobile app data by genre, price, size, currency, and user rating.
- 🤖 **LLM-Based Analysis**: Uses a chatbot to analyze app descriptions and provide insights into user engagement.
- 📈 **App Success Prediction**: Fine-tunes BERT to classify whether an app is likely to be successful.
- 📉 **Failure Explanation**: Uses **Gemma-2-9B** to analyze and explain why an app might fail or success.
- 🏗 **Streamlit UI**: Provides an interactive web-based dashboard for seamless user experience.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- pip
- PyTorch
- Streamlit
- Transformers (Hugging Face)
- Pandas

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/MobileAppStore-LLM-Analytics.git
cd MobileAppStore-LLM-Analytics
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run model.py
```

### App Interface

1. **Home**: Filter and explore app store data interactively.
2. **Data Explorer**: View dataset summary, statistics, and visualizations.
3. **App Analysis**:
   - Select an app.
   - Generate an AI-powered description.
   - Predict app success using fine-tuned **BERT**.
   - Receive failure explanation from **Gemma-2-9B**.

## Model Details

### Fine-Tuned BERT for Classification

- Tokenizer: `BertTokenizer`
- Model: `BertForSequenceClassification`
- Training: Fine-tuned on app store data to classify app success.

### Gemma-2-9B for Failure Explanation

- Provides insights into why an app might fail based on metadata and user feedback.
- Analyzes user sentiment and suggests improvements.

## Dataset

The project uses Apple App Store datasets:

- `AppleStore.csv`: Contains metadata about apps.
- `appleStore_description.csv`: Contains app descriptions.

## Contribution

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit changes and push to GitHub.
4. Create a pull request.

## License

This project is licensed under the MIT License. See `LICENSE` for details.



