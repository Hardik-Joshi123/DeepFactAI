import os
import json

# Create sample models directory structure
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

# Sample LSTM config
lstm_config = {
    "vocab_size": 10000,
    "embedding_dim": 300,
    "hidden_dim": 128,
    "output_dim": 2,
    "n_layers": 2,
    "bidirectional": True,
    "dropout": 0.3,
    "max_length": 128
}

# Sample BERT config
bert_config = {
    "model_name": "bert-base-uncased",
    "num_labels": 2,
    "max_length": 512,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1
}

# Save configs
with open(os.path.join(models_dir, 'lstm_config.json'), 'w') as f:
    json.dump(lstm_config, f, indent=2)

with open(os.path.join(models_dir, 'bert_config.json'), 'w') as f:
    json.dump(bert_config, f, indent=2)

print("Model configurations created successfully!")
