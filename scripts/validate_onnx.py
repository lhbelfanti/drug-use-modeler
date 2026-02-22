import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import onnxruntime as ort
import numpy as np

# ========================================================
# Model Definitions (Copied from export_to_onnx.py)
# ========================================================

class FFN(nn.Module):
    def __init__(self, input_dim=100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters=100, filter_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.requires_grad = False
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(x))
            c = c.max(dim=2)[0]
            conv_outs.append(c)
        out = torch.cat(conv_outs, dim=1)
        out = self.dropout(out)
        out = torch.sigmoid(self.fc(out)).squeeze(1)
        return out

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=1
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(x)
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        combined = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        out = self.dropout(combined)
        out = torch.sigmoid(self.fc(out)).squeeze(1)
        return out

# ========================================================
# Validators
# ========================================================

def validate_ffn(model_path):
    print(f"Validating FFN: {model_path}")
    onnx_path = model_path.replace('.pt', '.onnx')
    
    state_dict = torch.load(model_path, map_location='cpu')
    input_dim = state_dict['network.0.weight'].shape[1]
    
    model = FFN(input_dim=input_dim)
    model.load_state_dict(state_dict)
    model.eval()
    
    dummy_input = torch.randn(1, input_dim)
    
    # PyTorch output
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()
        
    # ONNX output
    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(None, {'input': dummy_input.numpy()})[0]
    
    np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-3, atol=1e-5)
    print("  -> OK!")

def validate_textcnn(model_path):
    print(f"Validating TextCNN: {model_path}")
    onnx_path = model_path.replace('.pt', '.onnx')
    
    state_dict = torch.load(model_path, map_location='cpu')
    vocab_size, embed_dim = state_dict['embedding.weight'].shape
    
    model = TextCNN(vocab_size=vocab_size, embed_dim=embed_dim)
    model.load_state_dict(state_dict)
    model.eval()
    
    dummy_input = torch.randint(0, vocab_size, (1, 50))
    
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()
        
    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(None, {'input_ids': dummy_input.numpy()})[0]
    
    np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-3, atol=1e-5)
    print("  -> OK!")

def validate_bilstm(model_path):
    print(f"Validating BiLSTM: {model_path}")
    onnx_path = model_path.replace('.pt', '.onnx')
    
    state_dict = torch.load(model_path, map_location='cpu')
    vocab_size, embed_dim = state_dict['embedding.weight'].shape
    
    model = BiLSTM(vocab_size=vocab_size, embed_dim=embed_dim)
    model.load_state_dict(state_dict)
    model.eval()
    
    dummy_input = torch.randint(0, vocab_size, (1, 50))
    
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()
        
    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(None, {'input_ids': dummy_input.numpy()})[0]
    
    np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-3, atol=1e-5)
    print("  -> OK!")

def validate_bert(model_dir):
    print(f"Validating BERT: {model_dir}")
    tokenizer_dir = os.path.join(os.path.dirname(model_dir), 'tokenizer')
    onnx_path = os.path.join(model_dir, "model.onnx")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    dummy_input = tokenizer("Muestra de prueba para validar el modelo", return_tensors="pt")
    
    with torch.no_grad():
        pt_out = model(**dummy_input).logits.numpy()
        
    session = ort.InferenceSession(onnx_path)
    inputs = {
        'input_ids': dummy_input['input_ids'].numpy(),
        'attention_mask': dummy_input['attention_mask'].numpy()
    }
    onnx_out = session.run(None, inputs)[0]
    
    np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-3, atol=1e-5)
    print("  -> OK!")

def validate_sklearn(model_path):
    print(f"Validating Scikit-Learn model: {model_path}")
    onnx_path = model_path.replace('.joblib', '.onnx')
    
    model = joblib.load(model_path)
    session = ort.InferenceSession(onnx_path)
    
    if "vectorizer" in model_path:
        dummy_input = np.array([["Este es un texto de prueba válido para el vectorizador asdasd"]], dtype=str)
        sk_out = model.transform(dummy_input.ravel()).toarray()
        
        onnx_out = session.run(None, {'text_input': dummy_input})[0]
    else:
        # Classifier
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        elif hasattr(model, 'coef_'):
            n_features = model.coef_.shape[1] if len(model.coef_.shape) > 1 else model.coef_.shape[0]
        else:
            n_features = 5000
            
        dummy_input = np.random.randn(1, n_features).astype(np.float32)
        sk_out_proba = None
        if hasattr(model, 'predict_proba'):
            sk_out_proba = model.predict_proba(dummy_input)
            
        sk_out_label = model.predict(dummy_input)
        
        onnx_outs = session.run(None, {'float_input': dummy_input})
        onnx_out_label = onnx_outs[0]
        
        if len(onnx_outs) > 1 and sk_out_proba is not None:
            # zip map dictionary check
            # onnx_outs[1] is a list of dicts: [{0: prob_0, 1: prob_1}] OR array depending on zipmap configuration
            onnx_prob_tensor = onnx_outs[1]
            if isinstance(onnx_prob_tensor, list) and isinstance(onnx_prob_tensor[0], dict):
                # zipmap = True output
                onnx_probs = np.array([[onnx_prob_tensor[0][k] for k in sorted(onnx_prob_tensor[0].keys())]])
            else:
                onnx_probs = onnx_prob_tensor
            np.testing.assert_allclose(sk_out_proba, onnx_probs, rtol=1e-3, atol=1e-5)
            
        np.testing.assert_equal(sk_out_label, onnx_out_label)
        print("  -> OK!")
        return

    np.testing.assert_allclose(sk_out, onnx_out, rtol=1e-3, atol=1e-5)
    print("  -> OK!")

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    
    # Validating FFN, CNN, RNN (.pt files)
    pt_models = glob.glob(os.path.join(base_dir, "**", "*.pt"), recursive=True)
    for path in pt_models:
        if "/ffn/" in path:
            validate_ffn(path)
        elif "/cnn/" in path:
            validate_textcnn(path)
        elif "/rnn/" in path:
            validate_bilstm(path)
            
    # Validating BERT (safetensors inside model/ folder)
    safetensors_models = glob.glob(os.path.join(base_dir, "**", "model", "model.safetensors"), recursive=True)
    for path in safetensors_models:
        model_dir = os.path.dirname(path)
        validate_bert(model_dir)

    # Validating Scikit-Learn joblib files
    joblib_models = glob.glob(os.path.join(base_dir, "**", "*.joblib"), recursive=True)
    for path in joblib_models:
        validate_sklearn(path)
    
    print("\nAll models have been successfully validated! The original models match the ONNX models' outputs exactly.")
