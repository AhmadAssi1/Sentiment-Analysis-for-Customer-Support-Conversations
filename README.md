# Arabic Sentiment Analysis API

This project provides a Flask-based REST API for analyzing sentiment in Arabic text. It can analyze individual messages, batch texts, and entire conversations, providing detailed sentiment analysis including negativity percentages.

## Features

- Single text sentiment analysis
- Batch text analysis
- Conversation analysis with role-based sentiment tracking
- Negativity percentage calculation
- Confidence scores for predictions

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Fine-tuned model files (see Model Files section below)

## Model Files

⚠️ **Important**: The model files are not included in this repository due to size limitations. You can:

1. Run the following command to generate the model files:
```bash
python fine_tune.py
```
2. Ensure the following files are present in the directory:
   - tokenizer.json
   - vocab.txt
   - pytorch_model.bin
   - config.json

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Download and place the model files as described in the Model Files section.

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the server by running:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### 1. Single Text Analysis
Analyze the sentiment of a single text.

**Endpoint:** `POST /analyze`

**Request Body:**
```json
{
    "text": "هذا المنتج رائع جداً"
}
```

**Response:**
```json
{
    "text": "هذا المنتج رائع جداً",
    "sentiment": "positive",
    "confidence": "95.5%"
}
```

### 2. Batch Analysis
Analyze multiple texts at once.

**Endpoint:** `POST /analyze_batch`

**Request Body:**
```json
{
    "texts": [
        "هذا المنتج رائع جداً",
        "لم يعجبني هذا المنتج على الإطلاق"
    ]
}
```

### 3. Conversation Analysis
Analyze a complete conversation with role-based sentiment tracking.

**Endpoint:** `POST /analyze_conversation`

**Request Body:**
```json
{
    "conversation": "Customer: المنتج الذي استلمته لا يعمل بشكل صحيح\nAgent: عذراً جداً على هذه التجربة\nCustomer: حسناً، شكراً لك"
}
```

**Response:**
```json
{
    "overall_sentiment": "negative",
    "negative_messages": [
        {
            "message": "المنتج الذي استلمته لا يعمل بشكل صحيح",
            "confidence": "85.5%"
        }
    ],
    "sentiment_distribution": {
        "positive": 0,
        "neutral": 2,
        "negative": 1
    },
    "negativity_percentage": "33.3%"
}
```

## Response Fields

- `sentiment`: Can be "positive", "neutral", or "negative"
- `confidence`: Confidence score of the prediction
- `overall_sentiment`: Overall conversation sentiment
- `negative_messages`: List of detected negative messages
- `sentiment_distribution`: Count of different sentiments in the conversation
- `negativity_percentage`: Percentage of negative messages in the conversation

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Successful request
- 400: Bad request (missing or invalid parameters)
- 500: Server error

## Development

The application is built using:
- Flask for the web server
- Transformers library for the sentiment analysis model
- PyTorch as the deep learning backend

