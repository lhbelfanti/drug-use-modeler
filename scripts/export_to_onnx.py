import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

# ========================================================
# Model Definitions
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
        # Pre-trained embedding layer (frozen)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.requires_grad = False  # Freeze embeddings
        
        # Conv1D filters at different window sizes
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
        # Pre-trained embedding (frozen)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.requires_grad = False
        
        # Bidirectional LSTM
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
# Exporters
# ========================================================

def export_ffn(model_path):
    print(f"Exporting FFN: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    # Determine input_dim from weights
    input_dim = state_dict['network.0.weight'].shape[1]
    
    model = FFN(input_dim=input_dim)
    model.load_state_dict(state_dict)
    model.eval()
    
    dummy_input = torch.randn(1, input_dim)
    output_path = model_path.replace('.pt', '.onnx')
    
    if os.path.exists(output_path):
        print(f"  -> Already exists {output_path}")
        return
        
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=14
    )
    print(f"  -> Saved to {output_path}")

def export_textcnn(model_path):
    print(f"Exporting TextCNN: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    vocab_size, embed_dim = state_dict['embedding.weight'].shape
    
    model = TextCNN(vocab_size=vocab_size, embed_dim=embed_dim)
    model.load_state_dict(state_dict)
    model.eval()
    
    dummy_input = torch.randint(0, vocab_size, (1, 50))  # batch=1, seq_len=50
    output_path = model_path.replace('.pt', '.onnx')
    
    if os.path.exists(output_path):
        print(f"  -> Already exists {output_path}")
        return
        
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=['input_ids'], output_names=['output'],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}, 'output': {0: 'batch_size'}},
        opset_version=14
    )
    print(f"  -> Saved to {output_path}")

def export_bilstm(model_path):
    print(f"Exporting BiLSTM: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    vocab_size, embed_dim = state_dict['embedding.weight'].shape
    
    model = BiLSTM(vocab_size=vocab_size, embed_dim=embed_dim)
    model.load_state_dict(state_dict)
    model.eval()
    
    dummy_input = torch.randint(0, vocab_size, (1, 50))
    output_path = model_path.replace('.pt', '.onnx')
    
    if os.path.exists(output_path):
        print(f"  -> Already exists {output_path}")
        return
        
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=['input_ids'], output_names=['output'],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}, 'output': {0: 'batch_size'}},
        opset_version=14
    )
    print(f"  -> Saved to {output_path}")

def export_bert(model_dir):
    print(f"Exporting BERT: {model_dir}")
    tokenizer_dir = os.path.join(os.path.dirname(model_dir), 'tokenizer')
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    dummy_input = tokenizer("Muestra de prueba", return_tensors="pt")
    
    output_path = os.path.join(model_dir, "model.onnx")
    
    if os.path.exists(output_path):
        print(f"  -> Already exists {output_path}")
        return
        
    # We only pass input_ids and attention_mask to typical BERT implementations
    inputs = (dummy_input['input_ids'], dummy_input['attention_mask'])
    
    torch.onnx.export(
        model, inputs, output_path,
        input_names=['input_ids', 'attention_mask'], output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14
    )
    print(f"  -> Saved to {output_path}")

def export_sklearn(model_path):
    print(f"Exporting Scikit-Learn model: {model_path}")
    output_path = model_path.replace('.joblib', '.onnx')

    if os.path.exists(output_path):
        print(f"  -> Already exists {output_path}")
        return

    model = joblib.load(model_path)
    
    if "vectorizer" in model_path:
        initial_type = [('text_input', StringTensorType([None, 1]))]
        options = None
    else:
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        elif hasattr(model, 'coef_'):
            n_features = model.coef_.shape[1] if len(model.coef_.shape) > 1 else model.coef_.shape[0]
        else:
            n_features = 5000
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        options = None
        if hasattr(model, 'predict_proba'):
            options = {id(model): {'zipmap': False}}
        
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"  -> Saved to {output_path}")


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    
    # Export FFN, CNN, RNN (.pt files)
    pt_models = glob.glob(os.path.join(base_dir, "**", "*.pt"), recursive=True)
    
    for path in pt_models:
        if "/ffn/" in path:
            export_ffn(path)
        elif "/cnn/" in path:
            export_textcnn(path)
        elif "/rnn/" in path:
            export_bilstm(path)
            
    # Export BERT (safetensors inside model/ folder)
    safetensors_models = glob.glob(os.path.join(base_dir, "**", "model", "model.safetensors"), recursive=True)
    
    for path in safetensors_models:
        model_dir = os.path.dirname(path)
        export_bert(model_dir)

    # Export Scikit-Learn joblib files
    joblib_models = glob.glob(os.path.join(base_dir, "**", "*.joblib"), recursive=True)
    for path in joblib_models:
        export_sklearn(path)
