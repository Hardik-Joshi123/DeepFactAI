"""
Configuration file for Fake News Detection System
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///fake_news.db')

# Model Configuration
MODELS = {
    'lstm': {
        'path': 'models/lstm_model.pth',
        'config_path': 'models/lstm_config.json',
        'enabled': True,
        'confidence_threshold': 0.7
    },
    'bert': {
        'path': 'models/bert_model',
        'config_path': 'models/bert_config.json',
        'enabled': True,
        'confidence_threshold': 0.75
    }
}

# API Configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
API_DEBUG = os.getenv('API_DEBUG', 'False').lower() == 'true'

# CORS Configuration
CORS_ORIGINS = [
    'http://localhost:3000',
    'http://localhost:8000',
    'http://127.0.0.1:3000',
    'http://127.0.0.1:8000',
]

if os.getenv('ENVIRONMENT') == 'production':
    CORS_ORIGINS.extend([
        'https://yourdomain.com',
        'https://api.yourdomain.com',
    ])

# Text Processing Configuration
MAX_TEXT_LENGTH = 5000
MIN_TEXT_LENGTH = 10

# Tokenizer Configuration
TOKENIZER_MAX_LENGTH = 512
TOKENIZER_MODEL = 'bert-base-uncased'

# Training Configuration
TRAINING = {
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'early_stopping_patience': 3
}

# Explainability Configuration
EXPLAINABILITY = {
    'method': 'lime',  # or 'shap'
    'num_features': 10,
    'num_samples': 1000
}

# Cache Configuration
CACHE_ENABLED = True
CACHE_TTL = 3600  # 1 hour

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Feature Flags
FEATURES = {
    'explain_predictions': True,
    'model_comparison': True,
    'ensemble_predictions': True,
    'sample_news': True,
    'model_stats': True
}
