from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load the fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

def classify_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = predictions.argmax().item()
    
    # Map predictions to labels
    labels = ['negative', 'neutral', 'positive']
    confidence = predictions[0][predicted_class].item()
    
    return {
        'text': text,
        'sentiment': labels[predicted_class],
        'confidence': f"{confidence:.2%}"
    }

def analyze_conversation(conversation_text):
    # Split the conversation into messages
    messages = conversation_text.strip().split('\n')
    
    # Lists to store sentiments and negative customer messages
    message_sentiments = []
    negative_customer_messages = []
    
    # Analyze each message
    for message in messages:
        if not message.strip():  # Skip empty lines
            continue
            
        # Assuming format: "Role: Message"
        try:
            role, content = message.split(':', 1)
            role = role.strip().lower()
            content = content.strip()
        except ValueError:
            continue
            
        # Analyze sentiment
        sentiment_result = classify_sentiment(content)
        message_sentiments.append(sentiment_result['sentiment'])
        
        # Track negative customer messages
        if role == 'customer' and sentiment_result['sentiment'] == 'negative':
            negative_customer_messages.append({
                'message': content,
                'confidence': sentiment_result['confidence']
            })
    
    # Calculate overall sentiment
    if not message_sentiments:
        return {
            'overall_sentiment': 'neutral',
            'negative_messages': [],
            'explanation': 'No valid messages found',
            'negativity_percentage': '0%'
        }
    
    # Count sentiments
    sentiment_counts = {
        'positive': message_sentiments.count('positive'),
        'neutral': message_sentiments.count('neutral'),
        'negative': message_sentiments.count('negative')
    }
    
    # Calculate negativity percentage
    total_messages = len(message_sentiments)
    negativity_percentage = (sentiment_counts['negative'] / total_messages) * 100
    
    # Determine overall sentiment
    max_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
    
    # If there's a significant number of negative messages, consider the conversation negative
    if sentiment_counts['negative'] >= len(message_sentiments) * 0.3:  # 30% threshold
        overall_sentiment = 'negative'
    else:
        overall_sentiment = max_sentiment
    
    return {
        'overall_sentiment': overall_sentiment,
        'negative_messages': negative_customer_messages,
        'sentiment_distribution': sentiment_counts,
        'negativity_percentage': f"{negativity_percentage:.1f}%"
    }

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Please provide text in the request body'
            }), 400
            
        text = data['text']
        result = classify_sentiment(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Please provide an array of texts in the request body'
            }), 400
            
        texts = data['texts']
        results = [classify_sentiment(text) for text in texts]
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/analyze_conversation', methods=['POST'])
def analyze_conversation_endpoint():
    try:
        data = request.get_json()
        
        if not data or 'conversation' not in data:
            return jsonify({
                'error': 'Please provide a conversation in the request body'
            }), 400
            
        conversation = data['conversation']
        result = analyze_conversation(conversation)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def home():
    return '''
    <h1>Arabic Sentiment Analysis API</h1>
    <h2>Usage:</h2>
    <h3>Single Text Analysis:</h3>
    <pre>
    POST /analyze
    Content-Type: application/json
    
    {
        "text": "هذا المنتج رائع جداً"
    }
    </pre>
    
    <h3>Batch Analysis:</h3>
    <pre>
    POST /analyze_batch
    Content-Type: application/json
    
    {
        "texts": [
            "هذا المنتج رائع جداً",
            "لم يعجبني هذا المنتج على الإطلاق"
        ]
    }
    </pre>

    <h3>Conversation Analysis:</h3>
    <pre>
    POST /analyze_conversation
    Content-Type: application/json
    
    {
        "conversation": "Customer: المنتج الذي استلمته لا يعمل بشكل صحيح\\nAgent: عذراً جداً على هذه التجربة\\nCustomer: حسناً، شكراً لك"
    }
    </pre>
    '''

if __name__ == '__main__':
    app.run(debug=True) 