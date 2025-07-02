import torch
import torch.nn.functional as F
from torch import no_grad
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torchviz
import torch.nn as nn
# import tokenizer
 


# class CNN(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, sequence_length):

#         super().__init__()
#         # Embedding layer
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
#         self.embedding.weight.requires_grad = False  # Freeze embeddings
#         #fixedy
#         # self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=3)
#         self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3)

#         self.pool = nn.MaxPool1d(kernel_size=2)


#         # Calculate input size after flattening embeddings
#         input_size = sequence_length * embedding_dim  # 20 * 50 = 1000

#         # MLP layers
#         self.fc1 = nn.Linear(9550, 4)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(4, 3)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(3, 3)
#         self.relu3 = nn.ReLU()
#         self.out = nn.Linear(3, 2)
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, X):
#         # X should be LongTensor (token indices)
#         embedded = self.embedding(X)  # (batch_size, seq_len, embedding_dim)
#         flattened = embedded.view(embedded.size(0), -1)  # (batch_size, 1000)

#         out = self.fc1(flattened)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         out = self.relu3(out)
#         out = self.out(out)
#         log_probs = self.log_softmax(out)
#         return log_probs
#     def forward(self, x):
#         x = self.embedding(x).permute(0, 2, 1)
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = x.view(x.size(0), -1)
# #         return self.fc(x)
# class CNN(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, sequence_length, emb_matrix):
#         super().__init__()

#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
#         self.embedding.weight.requires_grad = False  # Freeze embeddings

#         self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3)
#         self.pool = nn.MaxPool1d(kernel_size=2)

#         # Calculate conv output length for Linear layer
#         conv_output_len = (sequence_length - 3 + 1) // 2  # Using kernel_size=3, pool=2
#         self.fc = nn.Linear(64 * conv_output_len, 2)
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         x = self.embedding(x).permute(0, 2, 1)         # Shape: (batch_size, embedding_dim, seq_len)
#         x = self.pool(torch.relu(self.conv1(x)))       # Shape: (batch_size, 64, conv_output_len)
#         x = x.view(x.size(0), -1)                      # Flatten
#         x = self.fc(x)
#         return self.log_softmax(x)

# class CNN(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, sequence_length, emb_matrix):
#         super().__init__()

#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
#         self.embedding.weight.requires_grad = False

#         self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3)
#         self.pool = nn.MaxPool1d(kernel_size=2)

#         conv_output_len = (sequence_length - 3 + 1) // 2
#         flattened_dim = 64 * conv_output_len

#         self.fc2 = nn.Linear(flattened_dim, 3)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(3, 3)
#         self.relu3 = nn.ReLU()
#         self.out = nn.Linear(3, 2)
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         x = self.embedding(x).permute(0, 2, 1)
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = x.view(x.size(0), -1)
#         x = self.relu2(self.fc2(x))
#         x = self.relu3(self.fc3(x))
#         x = self.out(x)
#         return self.log_softmax(x)
# class CNN(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, sequence_length, emb_matrix):
#         super().__init__()

#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
#         self.embedding.weight.requires_grad = False

#         # Must match the original model's conv setup
#         self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3)
#         self.pool = nn.MaxPool1d(kernel_size=2)

#         # This must be 4 if that's what the original saved model used
#         self.fc2 = nn.Linear(4, 3)
#         self.fc3 = nn.Linear(3, 3)
#         self.out = nn.Linear(3, 2)
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         x = self.embedding(x)           # (batch, seq_len, embed_dim)
#         x = x.permute(0, 2, 1)          # (batch, embed_dim, seq_len)
#         x = self.conv1(x)               # (batch, out_channels=1, seq_len-2)
#         x = self.pool(torch.relu(x))    # (batch, 1, (seq_len-2)//2)
#         x = x.view(x.size(0), -1)       # Flatten -> (batch, 4) if matched

#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = self.out(x)
#         return self.log_softmax(x)

# import torch
# import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length, emb_matrix):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embedding.weight.requires_grad = False  # Freeze embeddings

        # Conv + Pool
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.fc1 = None  # Delay FC1 creation based on flattened shape

        # Other layers
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 3)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(3, 3)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(3, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        embedded = self.embedding(X)  # (batch_size, seq_len, emb_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, emb_dim, seq_len)

        convolved = self.pool(self.conv1(embedded))  # (batch_size, 64, reduced_seq_len)
        flattened = convolved.view(convolved.size(0), -1)

        print("CNN RUNTIME shape before fc1:", flattened.shape)

        if self.fc1 is None:
            self.fc1 = nn.Linear(flattened.size(1), 4).to(flattened.device)

        out = self.fc1(flattened)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.out(out)
        log_probs = self.log_softmax(out)
        return log_probs

# class LSTM(nn.Module):

#     def __init__(self, vocab_size, embedding_dim, sequence_length,emb_matrix):
#         super().__init__()

#         # Embedding layer
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
#         self.embedding.weight.requires_grad = False  # Freeze embeddings

#         # Calculate input size after flattening
#         input_size = sequence_length * embedding_dim  # 20 * 50 = 1000

#         # Sequential MLP
#         self.net = nn.Sequential(
#             nn.Linear(9550, 4),  # Changed from vocab_size to input_size
#             nn.ReLU(),
#             nn.Linear(4, 3),
#             nn.ReLU(),
#             nn.Linear(3, 3),
#             nn.ReLU(),
#             nn.Linear(3, 2),
#             nn.LogSoftmax(dim=1)
#         )

#     def forward(self, X):
#         # X should be LongTensor (token indices)
#         embedded = self.embedding(X)  # (batch_size, seq_len, embedding_dim)
#         flattened = embedded.view(embedded.size(0), -1)  # (batch_size, 1000)
#         log_probs = self.net(flattened)
#         return log_probs

# class LSTM(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, sequence_length, emb_matrix):
#         super().__init__()

#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
#         self.embedding.weight.requires_grad = False

#         self.fc1 = nn.Linear(embedding_dim * sequence_length, 4)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(4, 3)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(3, 3)
#         self.relu3 = nn.ReLU()
#         self.out = nn.Linear(3, 2)
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, X):
#         embedded = self.embedding(X)
#         flattened = embedded.view(embedded.size(0), -1)
#         out = self.fc1(flattened)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         out = self.relu3(out)
#         out = self.out(out)
#         return self.log_softmax(out)
# class LSTM(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, sequence_length, emb_matrix):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
#         self.embedding.weight.requires_grad = False  # Freeze embeddings

#         flattened_size = sequence_length * embedding_dim  # 191 * 50 = 9550 for example

#         # Match the exact layer structure used during training
#         self.fc1 = nn.Linear(flattened_size, 4)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(4, 3)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(3, 3)
#         self.relu3 = nn.ReLU()
#         self.out = nn.Linear(3, 2)
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, X):
#         embedded = self.embedding(X)                     # (batch_size, seq_len, embedding_dim)
#         flattened = embedded.view(embedded.size(0), -1)  # (batch_size, flattened_size)
#         out = self.fc1(flattened)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         out = self.relu3(out)
#         out = self.out(out)
#         return self.log_softmax(out)



class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length, emb_matrix):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embedding.weight.requires_grad = False  # Freeze embeddings

        # Delay defining net until we know flattened shape
        self.net = None

    def forward(self, X):
        # X: (batch_size, seq_len)
        embedded = self.embedding(X)  # (batch_size, seq_len, embedding_dim)

        # Flatten the embedding
        flattened = embedded.view(embedded.size(0), -1)  # (batch_size, seq_len * emb_dim)
       

        # Define the net dynamically based on input size
        if self.net is None:
            input_dim = flattened.size(1)
            self.net = nn.Sequential(
                nn.Linear(input_dim, 4),
                nn.ReLU(),
                nn.Linear(4, 3),
                nn.ReLU(),
                nn.Linear(3, 3),
                nn.ReLU(),
                nn.Linear(3, 2),
                nn.LogSoftmax(dim=1)
            ).to(flattened.device)

        log_probs = self.net(flattened)
        return log_probs

# class RNN(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, sequence_length, emb_matrix):

#         super().__init__()
#         # Embedding layer
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
#         self.embedding.weight.requires_grad = False  # Freeze embeddings

#         # Calculate input size after flattening embeddings
#         input_size = sequence_length * embedding_dim  # 20 * 50 = 1000

#         # MLP layers
#         self.fc1 = nn.Linear(9550, 4)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(4, 3)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(3, 3)
#         self.relu3 = nn.ReLU()
#         self.out = nn.Linear(3, 2)
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, X):
#         # X should be LongTensor (token indices)
#         embedded = self.embedding(X)  # (batch_size, seq_len, embedding_dim)
#         flattened = embedded.view(embedded.size(0), -1)  # (batch_size, 1000)

#         out = self.fc1(flattened)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         out = self.relu3(out)
#         out = self.out(out)
#         log_probs = self.log_softmax(out)
#         return log_probs
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length, emb_matrix):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embedding.weight.requires_grad = False  # Freeze embeddings

        # Placeholder for dynamically created fc1
        self.fc1 = None

        # Rest of the network
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 3)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(3, 3)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(3, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        embedded = self.embedding(X)  # (batch_size, seq_len, emb_dim)
        flattened = embedded.view(embedded.size(0), -1)  # (batch_size, seq_len * emb_dim)
        print("RUNTIME shape before fc1:", flattened.shape)

        # Dynamically create fc1 layer
        if self.fc1 is None:
            input_dim = flattened.size(1)
            self.fc1 = nn.Linear(input_dim, 4).to(flattened.device)

        out = self.fc1(flattened)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.out(out)
        log_probs = self.log_softmax(out)
        return log_probs

# STREAMLIT ML CLASSIFICATION APP - DUAL MODEL SUPPORT
# =====================================================

import docx2txt
import PyPDF2

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn.functional as F
#from keras.preprocessing.sequence import pad_sequences 
import pickle
# with open("emb_matrix.pkl", "rb") as f:
#     emb_matrix = pickle.load(f)

import pickle

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ✅ Load embedding matrix
with open("emb_matrix.pkl", "rb") as f:
    emb_matrix = pickle.load(f)



# Page Configuration
st.set_page_config(
    page_title="ML Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_file(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return "Unsupported file type."
    
# tokenizer = joblib.load('models/tokenizer.pkl')
# vocab_size = len(tokenizer.word_index) + 1

   
embedding_dim = 50
sequence_length = 191
# ============================================================================
# MODEL LOADING SECTION
# ============================================================================

@st.cache_resource
def load_models():
    
    models = {}
    
    try:
        # Load the main pipeline (SVM)
        try:
            models['svm'] = joblib.load('models/svm_model.pkl')
            models['svm_available'] = True
        except FileNotFoundError:
            models['svm_available'] = False
        
        # Load TF-IDF vectorizer
        try:
            models['vectorizer'] = joblib.load('models/tfidf_vectorizer.pkl')
            models['vectorizer_available'] = True
        except FileNotFoundError:
            models['vectorizer_available'] = False
        
        # Load Adaboost model
        try:
            models['adaboost'] = joblib.load('models/adaboost_model.pkl')
            models['adaboost_available'] = True
        except FileNotFoundError:
            models['adaboost_available'] = False
        
        # Load Decision tree model
        try:
            models['decision_tree'] = joblib.load('models/decision_tree_model.pkl')
            models['decision_tree_available'] = True
        except FileNotFoundError:
            models['decision_tree_available'] = False
            
    
        try:
           tokenizer = joblib.load('models/tokenizer.pkl')
           vocab_size = len(tokenizer.word_index) + 1
           st.success("✅ Tokenizer loaded successfully.")
        except Exception as e:
         st.error(f"❌ Error loading tokenizer: {e}")
         return None
        
        
        # try:
        #     import pickle
        #     with open("emb_matrix.pkl", "rb") as f:
        #       emb_matrix = pickle.load(f)
        # except FileNotFoundError:
         st.error("Embedding matrix file not found. Please generate 'emb_matrix.pkl'.")


        #     # Load CNN model
        # try:
        #     models['CNN'] = CNNassifier(vocab_size, embedding_dim, sequence_length).load_state_dict(torch.load('models/CNN.pkl', map_location='cpu'))
       
        #     models['CNN_available'] = True
        # except FileNotFoundError:
        #     models['CNN_available'] = False

        #     #Load LSTM model
        # try:
        #     models['LSTM'] = LSTMClassifier(vocab_size, embedding_dim, 128, sequence_length).load_state_dict(torch.load('models/LSTM.pkl', map_location='cpu'))
        #     models['LSTM_available'] = True
        # except FileNotFoundError:
        #     models['LSTM_available'] = False

        #     #Load RNN model
        # try:
        #     models['RNN'] = RNNClassifier(vocab_size, embedding_dim, 128, sequence_length).load_state_dict(torch.load('models/RNN.pkl', map_location='cpu'))
        #     models['RNN_available'] = True
        # except FileNotFoundError:
        #     models['RNN_available'] = False

        # Load CNN model
        try:
            cnn_model = CNN(vocab_size, embedding_dim, sequence_length, emb_matrix)
            cnn_model.load_state_dict(torch.load('models/CNN.pkl', map_location='cpu', weights_only=False))

            cnn_model.eval()
            models['CNN'] = cnn_model
            models['CNN_available'] = True
        except FileNotFoundError:
          models['CNN_available'] = False

# Load LSTM model
        try:
           lstm_model = LSTM(vocab_size, embedding_dim, sequence_length, emb_matrix)
           lstm_model.load_state_dict(torch.load('models/LSTM.pkl', map_location='cpu' , weights_only=False))
           lstm_model.eval()
           models['LSTM'] = lstm_model
           models['LSTM_available'] = True
        except FileNotFoundError:
            models['LSTM_available'] = False

# Load RNN model
        try:
            rnn_model = RNN(vocab_size, embedding_dim, sequence_length, emb_matrix)
            rnn_model.load_state_dict(torch.load('models/RNN.pkl', map_location='cpu' , weights_only=False))
            rnn_model.eval()
            models['RNN'] = rnn_model
            models['RNN_available'] = True
        except FileNotFoundError:
            models['RNN_available'] = False

        
        # Check if at least one complete setup is available
        pipeline_ready = models['svm_available']
        individual_ready = models['vectorizer_available'] and (models['adaboost_available'] or models['decision_tree_available'] or models['CNN_available'] or models['LSTM_available'] or models['RNN_available'])
        
        if not (pipeline_ready or individual_ready):
            st.error("No complete model setup found!")
            return None
        
        return models
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None



# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
#"fixedy"
# def predict_dl_model(model, text, tokenizer, sequence_length):
#     model.eval()
#     tokens = tokenizer.texts_to_sequences([text])
#     padded = torch.tensor(np.array(tokens), dtype=torch.long)
#     with torch.no_grad():
#         output = model(padded)
#         pred = (output >= 0.5).int().item()
#         prob = output.item()
    #return ("AI" if pred == 1 else "Human"), [1 - prob, prob]
def predict_dl_model(model, text, tokenizer, sequence_length):
    # import torch
    # import torch.nn.functional as F
    # from torch import no_grad
    # from tensorflow.keras.preprocessing.sequence import pad_sequences
    # import numpy as np

    model.eval()

    # Tokenize and pad
    tokens = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(tokens, maxlen=sequence_length, padding='post')
    input_tensor = torch.tensor(padded).long().to(next(model.parameters()).device)

    with no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    return pred, probs

def make_prediction(text, model_choice, models):
    """Make prediction using the selected model"""
    if models is None:
        return None, None
    
    try:
        prediction = None
        probabilities = None
        
        if model_choice == "svm" and models.get('svm_available'):
            # Use the complete pipeline (SVM)
            prediction = models['svm'].predict([text])[0]
            probabilities = models['svm'].predict_proba([text])[0]
            
        elif model_choice == "adaboost":
            if models.get('adaboost_available'):
                # Use pipeline for SVM
                prediction = models['adaboost'].predict([text])[0]
                probabilities = models['adaboost'].predict_proba([text])[0]
            # elif models.get('vectorizer_available') and models.get('adaboost_available'):
            #     # Use individual components
            #     X = models['vectorizer'].transform([text])
            #     prediction = models['logistic_regression'].predict(X)[0]
            #     probabilities = models['logistic_regression'].predict_proba(X)[0]
                
        elif model_choice == "decision_tree":
            if models.get('decision_tree_available'): # removed  and models.get('nb_available')
                # Use individual components for NB
                #X = models['vectorizer'].transform([text])
                prediction = models['decision_tree'].predict([text])[0]
                probabilities = models['decision_tree'].predict_proba([text])[0]
        elif model_choice == "CNN":
             if models.get('CNN_available'):
              prediction, probabilities = predict_dl_model(models['CNN'], text, tokenizer, sequence_length)

        elif model_choice == "LSTM":
             if models.get('LSTM_available'):
                prediction, probabilities = predict_dl_model(models['LSTM'], text, tokenizer, sequence_length)

        elif model_choice == "RNN":
             if models.get('RNN_available'):
                prediction, probabilities = predict_dl_model(models['RNN'], text, tokenizer, sequence_length)
        

        # elif model_choice == "CNN":
        #     if models.get('CNN_available'): # removed  and models.get('nb_available')
        #         # Use individual components for NB
        #         #X = models['vectorizer'].transform([text])
        #         prediction = models['CNN'].predict([text])[0]
        #         probabilities = models['CNN'].predict_proba([text])[0]

        # elif model_choice == "LSTM":
        #     if models.get('LSTM_available'):
                
        #         prediction = models['LSTM'].predict([text])[0]
        #         probabilities = models['LSTM'].predict_proba([text])[0]

        # elif model_choice == "RNN":
        #     if models.get('RNN_available'):
                
                # prediction = models['RNN'].predict([text])[0]
                # probabilities = models['RNN'].predict_proba([text])[0]
        
        if prediction is not None and probabilities is not None:
            # Convert to readable format
            class_names = ['Human', 'AI']
            prediction_label = class_names[prediction]
            return prediction_label, probabilities
        else:
            return None, None
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(f"Model choice: {model_choice}")
        st.error(f"Available models: {[k for k, v in models.items() if isinstance(v, bool) and v]}")
        return None, None

def get_available_models(models):
    """Get list of available models for selection"""
    available = []
    
    if models is None:
        return available
    
    if models.get('svm_available'):
        available.append(("svm", "📈 SVM (Pipeline)"))
    if models.get('adaboost_available'): # removed  and models.get('lr_available')
        available.append(("adaboost", "📈 Adaboost (Pipeline)"))
    
    if models.get('decision_tree_available'): #  and models.get('nb_available')
        available.append(("decision_tree", "🎯 Decision Tree (Pipeline)"))

    if models.get('CNN_available'): #  and models.get('nb_available')
        available.append(("CNN", "🎯 CNN (individual)"))

    if models.get('LSTM_available'): #  and models.get('nb_available')
        available.append(("LSTM", "🎯 LSTM (individual)"))

    if models.get('RNN_available'): #  and models.get('nb_available')
        available.append(("RNN", "🎯 RNN (individual)"))
    
    return available

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Choose what you want to do:")

page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Single Prediction", "📁 Batch Processing", "⚖️ Model Comparison", "📊 Model Info", "❓ Help"]
)

# Load models
models = load_models()

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 ML Text Classification App</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to your machine learning web application! This app demonstrates text analysis
    using multiple trained models: **SVM** , **Decision Tree**, **Adaboost**, **CNN**, **LSTM**, **RNN**.
    """)
    
    # App overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔮 Single Prediction
        - Enter text manually
        - Choose between models
        - Get instant predictions
        - See confidence scores
        """)
    
    with col2:
        st.markdown("""
        ### 📁 Batch Processing
        - Upload text files
        - Process multiple texts
        - Compare model performance
        - Download results
        """)
    
    with col3:
        st.markdown("""
        ### ⚖️ Model Comparison
        - Compare different models
        - Side-by-side results
        - Agreement analysis
        - Performance metrics
        """)
    
    # Model status
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            if models.get('svm_available'):
                st.info("**📈 SVM **\n✅ Pipeline Available")
            # elif models.get('adaboost_available') and models.get('vectorizer_available'):
            #     st.info("**📈 Adaboost**\n✅ Individual Components")
            else:
                st.warning("**📈 SVM**\n❌ Not Available")
        
        with col2:
            if models.get('adaboost_available'):
                st.info("**🎯 Adaboost**\n✅ Available")
            else:
                st.warning("**🎯 Adaboost**\n❌ Not Available")
        
        with col3:
            if models.get('decision_tree_available'):
                st.info("**🔤 Decsision Tree**\n✅ Available")
            else:
                st.warning("**🔤 Decsision Tree**\n❌ Not Available")

        with col4:
            if models.get('CNN_available'):
                st.info("**🔤 CNN**\n✅ Available")
            else:
                st.warning("**🔤 CNN**\n❌ Not Available")

        with col5:
            if models.get('LSTM_available'):
                st.info("**🔤 LSTM**\n✅ Available")
            else:
                st.warning("**🔤 LSTM**\n❌ Not Available")

        with col6:
            if models.get('RNN_available'):
                st.info("**🔤 RNN**\n✅ Available")
            else:
                st.warning("**🔤 RNN**\n❌ Not Available")
        
    else:
        st.error("❌ Models not loaded. Please check model files.")

# # ============================================================================
# # SINGLE PREDICTION PAGE
# # ============================================================================


# ============================================================================
# SINGLE PREDICTION PAGE
# ============================================================================
import docx2txt
from PyPDF2 import PdfReader


from PyPDF2 import PdfReader
import docx2txt

if page == "🔮 Single Prediction":
    st.header("🔮 Make a Single Prediction")
    st.markdown("Enter text below or upload a document (TXT, PDF, DOCX) to get prediction.")

    if models:
        available_models = get_available_models(models)

        if available_models:
            model_choice = st.selectbox(
                "Choose a model:",
                options=[model[0] for model in available_models],
                format_func=lambda x: next(name for key, name in available_models if key == x)
            )

            st.markdown("📄 **Or upload a document (TXT, PDF, DOCX):**")
            uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf', 'docx'])

            user_input = ""

            if uploaded_file:
                file_type = uploaded_file.name.split('.')[-1]

                if file_type == "txt":
                    user_input = str(uploaded_file.read(), "utf-8")

                elif file_type == "pdf":
                    pdf = PdfReader(uploaded_file)
                    user_input = "\n".join([page.extract_text() or "" for page in pdf.pages])

                elif file_type == "docx":
                    user_input = docx2txt.process(uploaded_file)

            # If no file, use manual input
            user_input = st.text_area(
                "✍️ Enter your text here:",
                value=user_input,
                placeholder="Type or paste your text here (e.g., product review, comment)...",
                height=200
            )

            if user_input:
                st.caption(f"Character count: {len(user_input)} | Word count: {len(user_input.split())}")

            if st.button("🚀 Predict", type="primary"):
                if user_input.strip():
                    with st.spinner("Analyzing text..."):
                        prediction, probabilities = make_prediction(user_input, model_choice, models)

                        if prediction and probabilities is not None:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                if prediction == "Human":
                                    st.success(f"🎯 Prediction: **{prediction} Text**")
                                else:
                                    st.error(f"🎯 Prediction: **{prediction} Text**")
                            with col2:
                                confidence = max(probabilities)
                                st.metric("Confidence", f"{confidence:.1%}")

                            st.subheader("📊 Prediction Probabilities")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("😞 Human", f"{probabilities[0]:.1%}")
                            with col2:
                                st.metric("😊 AI", f"{probabilities[1]:.1%}")

                            chart_df = pd.DataFrame({
                                "Label": ["Human", "AI"],
                                "Probability": probabilities
                            })
                            st.bar_chart(chart_df.set_index("Label"))

                        else:
                            st.error("Failed to make prediction.")
                else:
                    st.warning("Please enter or upload some text!")
        else:
            st.error("No models available for prediction.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# if page == "🔮 Single Prediction":
#     st.header("🔮 Make a Single Prediction")
#     st.markdown("Enter text below and select a model to get sentiment predictions.")
    
#     if models:
#         available_models = get_available_models(models)
        
#         if available_models:
#             # Model selection
#             model_choice = st.selectbox(
#                 "Choose a model:",
#                 options=[model[0] for model in available_models],
#                 #format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
#                 format_func=lambda x: next(name for key, name in available_models if key == x)
#             )
            
#             # Text input
#             user_input = st.text_area(
#                 "Enter your text here:",
#                 placeholder="Type or paste your text here (e.g., product review, feedback, comment)...",
#                 height=150
#             )
#             #File uploader (optional)
#             uploaded_file = st.file_uploader("📄 Or upload a document (TXT, PDF, DOCX):", type=['txt', 'pdf', 'docx'])

#             user_input = ""
#         if uploaded_file:
#           file_type = uploaded_file.type

#           if file_type == "text/plain":
#             user_input = str(uploaded_file.read(), "utf-8")

#           elif file_type == "application/pdf":
#             pdf = PdfReader(uploaded_file)
#             for page in pdf.pages:
#              user_input += page.extract_text() + "\n"

#           elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
#                user_input = docx2txt.process(uploaded_file)
#             # Character count
#         if user_input:
#                 st.caption(f"Character count: {len(user_input)} | Word count: {len(user_input.split())}")
            
#             # Example texts
#         with st.expander("📝 Try these example texts"):
#                 examples = [
#                     "This product is absolutely amazing! Best purchase I've made this year.",
#                     "Terrible quality, broke after one day. Complete waste of money.",
#                     "It's okay, nothing special but does the job.",
#                     "Outstanding customer service and fast delivery. Highly recommend!",
#                     "I love this movie! It's absolutely fantastic and entertaining."
#                 ]
                
#                 col1, col2 = st.columns(2)
#                 for i, example in enumerate(examples):
#                     with col1 if i % 2 == 0 else col2:
#                         if st.button(f"Example {i+1}", key=f"example_{i}"):
#                             st.session_state.user_input = example
#                             st.rerun()
            
#             # Use session state for user input
#         if 'user_input' in st.session_state:
#                 user_input = st.session_state.user_input
            
#             # Prediction button
#         if st.button("🚀 Predict", type="primary"):
#                 if user_input.strip():
#                     with st.spinner('Analyzing text...'):
#                         prediction, probabilities = make_prediction(user_input, model_choice, models)
                        
#                         if prediction and probabilities is not None:
#                             # Display prediction
#                             col1, col2 = st.columns([3, 1])
                            
#                             with col1:
#                                 if prediction == "Human":
#                                     st.success(f"🎯 Prediction: **{prediction} Text**")
#                                 else:
#                                     st.error(f"🎯 Prediction: **{prediction} Text**")
                            
#                             with col2:
#                                 confidence = max(probabilities)
#                                 st.metric("Confidence", f"{confidence:.1%}")
                            
#                             # Create probability chart
#                             st.subheader("📊 Prediction Probabilities")
                            
#                             # Detailed probabilities
#                             col1, col2 = st.columns(2)
#                             with col1:
#                                 st.metric("😞 Human", f"{probabilities[0]:.1%}")
#                             with col2:
#                                 st.metric("😊 AI", f"{probabilities[1]:.1%}")
                            
#                             # Bar chart
#                             class_names = ['Human', 'AI']
#                             prob_df = pd.DataFrame({
#                                 'Text': class_names,
#                                 'Probability': probabilities
#                             })
#                             st.bar_chart(prob_df.set_index('Text'), height=300)
                            
#                         else:
#                             st.error("Failed to make prediction")
#                 else:
#                     st.warning("Please enter some text to classify!")
#         else:
#             st.error("No models available for prediction.")
#     else:
#         st.warning("Models not loaded. Please check the model files.")

# ============================================================================
# BATCH PROCESSING PAGE
# ============================================================================
elif page == "📁 Batch Processing":
    st.header("📁 Upload File for Batch Processing")
    st.markdown("Upload a text file or CSV to process multiple texts at once.")
    
    if models:
        available_models = get_available_models(models)
        
        if available_models:
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'csv', 'pdf', 'docx'],
                help="Upload a .txt file (one text per line) or .csv file (text in first column)"
            )
            
            if uploaded_file:
                # Model selection
                model_choice = st.selectbox(
                    "Choose model for batch processing:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
                )
                
                # Process file
                if st.button("📊 Process File"):
                    try:
                        # Read file content
                        # if uploaded_file.type == "text/plain":
                        #     content = str(uploaded_file.read(), "utf-8")
                        #     texts = [line.strip() for line in content.split('\n') if line.strip()]
                        # else:  # CSV
                        #     df = pd.read_csv(uploaded_file)
                        #     texts = df.iloc[:, 0].astype(str).tolist()
                        
                        # if not texts:
                        #     st.error("No text found in file")
                        # else:
                        #     st.info(f"Processing {len(texts)} texts...")
                            
                            # Process all texts
                            results = []
                            progress_bar = st.progress(0)
                            
                            for i, text in enumerate(texts):
                                if text.strip():
                                    prediction, probabilities = make_prediction(text, model_choice, models)
                                    
                                    if prediction and probabilities is not None:
                                        results.append({
                                            'Text': text[:100] + "..." if len(text) > 100 else text,
                                            'Full_Text': text,
                                            'Prediction': prediction,
                                            'Confidence': f"{max(probabilities):.1%}",
                                            'Human_Prob': f"{probabilities[0]:.1%}",
                                            'AI_Prob': f"{probabilities[1]:.1%}"
                                        })
                                
                                progress_bar.progress((i + 1) / len(texts))
                            
                            if results:
                                # Display results
                                st.success(f"✅ Processed {len(results)} texts successfully!")
                                
                                results_df = pd.DataFrame(results)
                                
                                # Summary statistics
                                st.subheader("📊 Summary Statistics")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                human_count = sum(1 for r in results if r['Prediction'] == 'Human')
                                ai_count = len(results) - human_count
                                avg_confidence = np.mean([float(r['Confidence'].strip('%')) for r in results])
                                
                                with col1:
                                    st.metric("Total Processed", len(results))
                                with col2:
                                    st.metric("😊 Human", human_count)
                                with col3:
                                    st.metric("😞 AI", ai_count)
                                with col4:
                                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                                
                                # Results preview
                                st.subheader("📋 Results Preview")
                                st.dataframe(
                                    results_df[['Text', 'Prediction', 'Confidence']],
                                    use_container_width=True
                                )
                                
                                # Download option
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Full Results",
                                    data=csv,
                                    file_name=f"predictions_{model_choice}_{uploaded_file.name}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("No valid texts could be processed")
                                
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
            else:
                st.info("Please upload a file to get started.")
                
                # Show example file formats
                with st.expander("📄 Example File Formats"):
                    st.markdown("""
                    **Text File (.txt):**
                    ```
                    This product is amazing!
                    Terrible quality, very disappointed
                    Great service and fast delivery
                    ```
                    
                    **CSV File (.csv):**
                    ```
                    text,category
                    "Amazing product, love it!",review
                    "Poor quality, not satisfied",review
                    ```
                    """)
        else:
            st.error("No models available for batch processing.")
    else:
        st.warning("Models not loaded. Please check the model files.")


# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================
# ==
import docx2txt
from PyPDF2 import PdfReader

# elif page == "⚖️ Model Comparison":
#     st.header("⚖️ Compare Models")
#     st.markdown("Compare predictions from different models on the same text.")

#     if models:
#         available_models = get_available_models(models)

#         if len(available_models) >= 2:
#             # Option for text input or file upload
#             input_mode = st.radio("Choose input method:", ["✍️ Type/Paste Text", "📄 Upload File"])

#             comparison_text = ""

#             if input_mode == "✍️ Type/Paste Text":
#                 comparison_text = st.text_area(
#                     "Enter text to compare models:",
#                     placeholder="Enter or paste your text here...",
#                     height=150
#                 )
#             else:
#                 uploaded_file = st.file_uploader("Upload a document (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])
#                 if uploaded_file:
#                     file_type = uploaded_file.name.split('.')[-1].lower()
#                     if file_type == "txt":
#                         comparison_text = str(uploaded_file.read(), "utf-8")
#                     elif file_type == "pdf":
#                         pdf = PdfReader(uploaded_file)
#                         comparison_text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
#                     elif file_type == "docx":
#                         comparison_text = docx2txt.process(uploaded_file)
#                     else:
#                         st.error("Unsupported file format")

#             if st.button("📊 Compare All Models") and comparison_text.strip():
#                 st.subheader("🔍 Model Comparison Results")

#                 # Get predictions from all available models
#                 comparison_results = []

#                 for model_key, model_name in available_models:
#                     prediction, probabilities = make_prediction(comparison_text, model_key, models)

#                     if prediction and probabilities is not None:
#                         comparison_results.append({
#                             'Model': model_name,
#                             'Prediction': prediction,
#                             'Confidence': f"{max(probabilities):.1%}",
#                             'Human %': f"{probabilities[0]:.1%}",
#                             'AI %': f"{probabilities[1]:.1%}",
#                             'Raw_Probs': probabilities
#                         })

#                 if comparison_results:
#                     # Comparison table
#                     comparison_df = pd.DataFrame(comparison_results)
#                     st.table(comparison_df[['Model', 'Prediction', 'Confidence', 'Human %', 'AI %']])

#                     # Agreement analysis
#                     predictions = [r['Prediction'] for r in comparison_results]
#                     if len(set(predictions)) == 1:
#                         st.success(f"✅ All models agree: **{predictions[0]} text**")
#                     else:
#                         st.warning("⚠️ Models disagree on prediction")
#                         for result in comparison_results:
#                             model_name = result['Model'].split(' ')[1] if ' ' in result['Model'] else result['Model']
#                             st.write(f"- {model_name}: {result['Prediction']}")

#                     # Side-by-side probability charts
#                     st.subheader("📊 Detailed Probability Comparison")

#                     cols = st.columns(len(comparison_results))

#                     for i, result in enumerate(comparison_results):
#                         with cols[i]:
#                             model_name = result['Model']
#                             st.write(f"**{model_name}**")

#                             chart_data = pd.DataFrame({
#                                 'text': ['AI', 'Human'],
#                                 'Probability': result['Raw_Probs']
#                             })
#                             st.bar_chart(chart_data.set_index('text'))
#                 else:
#                     st.error("Failed to get predictions from models")

#         elif len(available_models) == 1:
#             st.info("Only one model available. Use Single Prediction page for detailed analysis.")
#         else:
#             st.error("No models available for comparison.")
#     else:
#         st.warning("Models not loaded. Please check the model files.")
if page == "⚖️ Model Comparison":
    st.header("⚖️ Compare Models")
    st.markdown("Compare predictions from different models on the same text.")
    
    if models:
        available_models = get_available_models(models)
        
        #if len(available_models) >= 2:
            # # Text input for comparison
            # comparison_text = st.text_area(
            #     "Enter text to compare models:",
            #     placeholder="Enter text to see how different models perform...",
            #     height=100)

        if len(available_models) >= 2:
            # Option for text input or file upload
            input_mode = st.radio("Choose input method:", ["✍️ Type/Paste Text", "📄 Upload File"])

            comparison_text = ""

            if input_mode == "✍️ Type/Paste Text":
                comparison_text = st.text_area(
                    "Enter text to compare models:",
                    placeholder="Enter or paste your text here...",
                    height=150
                )
            else:
                uploaded_file = st.file_uploader("Upload a document (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])
                if uploaded_file:
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    if file_type == "txt":
                        comparison_text = str(uploaded_file.read(), "utf-8")
                    elif file_type == "pdf":
                        pdf = PdfReader(uploaded_file)
                        comparison_text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
                    elif file_type == "docx":
                        comparison_text = docx2txt.process(uploaded_file)
                    else:
                        st.error("Unsupported file format")

            if st.button("📊 Compare All Models") and comparison_text.strip():
                st.subheader("🔍 Model Comparison Results")
            
            # if st.button("📊 Compare All Models") and comparison_text.strip():
            #     st.subheader("🔍 Model Comparison Results")
                
                # Get predictions from all available models
                comparison_results = []
                
                for model_key, model_name in available_models:
                    prediction, probabilities = make_prediction(comparison_text, model_key, models)
                    
                    if prediction and probabilities is not None:
                        comparison_results.append({
                            'Model': model_name,
                            'Prediction': prediction,
                            'Confidence': f"{max(probabilities):.1%}",
                            'Human %': f"{probabilities[0]:.1%}",
                            'AI %': f"{probabilities[1]:.1%}",
                            'Raw_Probs': probabilities
                        })
                
                if comparison_results:
                    # Comparison table
                    comparison_df = pd.DataFrame(comparison_results)
                    st.table(comparison_df[['Model', 'Prediction', 'Confidence', 'Human %', 'AI %']])
                    
                    #Agreement analysis
                    predictions = [r['Prediction'] for r in comparison_results]
                    if len(set(predictions)) == 1:
                        st.success(f"✅ All models agree: **{predictions[0]} text**")
                        
                    else:
                        st.warning("⚠️ Models disagree on prediction")
                        for result in comparison_results:
                            model_name = result['Model'].split(' ')[1] if ' ' in result['Model'] else result['Model']
                            st.write(f"- {model_name}: {result['Prediction']}")
                    
                    # Side-by-side probability charts
                    st.subheader("📊 Detailed Probability Comparison")
                    
                    cols = st.columns(len(comparison_results))
                    
                    for i, result in enumerate(comparison_results):
                        with cols[i]:
                            model_name = result['Model']
                            st.write(f"**{model_name}**")
                            
                            chart_data = pd.DataFrame({
                                'text': ['AI', 'Human'],
                                'Probability': result['Raw_Probs']
                            })
                            st.bar_chart(chart_data.set_index('text'))
                    
                else:
                    st.error("Failed to get predictions from models")
        
        elif len(available_models) == 1:
            st.info("Only one model available. Use Single Prediction page for detailed analysis.")
            
        else:
            st.error("No models available for comparison.")
    else:
        st.warning("Models not loaded. Please check the model files.")

elif page == "⚖️ Model Comparison":
    st.header("⚖️ Compare Models")
    st.markdown("Compare predictions from different models on the same text.")

    if models:
        available_models = get_available_models(models)

        if len(available_models) >= 2:
            comparison_text = None

            st.subheader("✍️ Paste Text or 📄 Upload Document")

            uploaded_file = st.file_uploader("Choose a file (optional)", type=['txt', 'pdf', 'docx'])
            if uploaded_file is not None:
                if uploaded_file.type == "text/plain":
                    comparison_text = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "application/pdf":
                    pdf_reader = PdfReader(uploaded_file)
                    comparison_text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    comparison_text = docx2txt.process(uploaded_file)

            if not comparison_text:
                comparison_text = st.text_area(
                    "Or paste your text below:",
                    placeholder="Paste text here for model comparison...",
                    height=150
                )

            if st.button("📊 Compare All Models") and comparison_text.strip():
                st.subheader("🔍 Model Comparison Results")
                comparison_results = []

                for model_key, model_name in available_models:
                    prediction, probabilities = make_prediction(comparison_text, model_key, models)

                    if prediction and probabilities is not None:
                        comparison_results.append({
                            'Model': model_name,
                            'Prediction': prediction,
                            'Confidence': f"{max(probabilities):.1%}",
                            'Human %': f"{probabilities[0]:.1%}",
                            'AI %': f"{probabilities[1]:.1%}",
                            'Raw_Probs': probabilities
                        })

                if comparison_results:
                    comparison_df = pd.DataFrame(comparison_results)
                    st.table(comparison_df[['Model', 'Prediction', 'Confidence', 'Human %', 'AI %']])

                    predictions = [r['Prediction'] for r in comparison_results]
                    if len(set(predictions)) == 1:
                        st.success(f"✅ All models agree: **{predictions[0]} text**")
                    else:
                        st.warning("⚠️ Models disagree on prediction")
                        for result in comparison_results:
                            model_name = result['Model'].split(' ')[1] if ' ' in result['Model'] else result['Model']
                            st.write(f"- {model_name}: {result['Prediction']}")

                    st.subheader("📊 Detailed Probability Comparison")
                    cols = st.columns(len(comparison_results))
                    for i, result in enumerate(comparison_results):
                        with cols[i]:
                            st.write(f"**{result['Model']}**")
                            chart_data = pd.DataFrame({
                                'text': ['AI', 'Human'],
                                'Probability': result['Raw_Probs']
                            })
                            st.bar_chart(chart_data.set_index('text'))
                else:
                    st.error("Failed to get predictions from models")

        elif len(available_models) == 1:
            st.info("Only one model available. Use Single Prediction page for detailed analysis.")
        else:
            st.error("No models available for comparison.")
    else:
        st.warning("Models not loaded. Please check the model files.")


# ============================================================================
# MODEL INFO PAGE
# ============================================================================

elif page == "📊 Model Info":
    st.header("📊 Model Information")
    
    if models:
        st.success("✅ Models are loaded and ready!")
        
        # Model details
        st.subheader("🔧 Available Models")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.markdown("""
            ### 📈 SVM model
            **Type:** pType: Machine Learning Classifier
            **Algorithm:** Support Vector Machine (SVM)
            **Features:** TF-IDF vectors (unigrams + bigrams)
            
            **Strengths:**
            - Fast prediction
            - Strong performance on high-dimensional sparse data (e.g., text)
            - Effective for binary classification
            - Handles sparse features well
            """)
 


        with col2:
            st.markdown("""
            
           ### 🎯 Decision Tree mode
                **Type:** Machine Learning Classifier
                **Algorithm:** Decision Tree
                **Features:** TF-IDF Vectors (Unigrams + Bigrams)

                **Strengths:**

            - Easy to interpret (rule-based)

                - Fast training and prediction

                - Handles non-linear data

               - Performs well with small datasets
                """)
        with col3:
            st.markdown("""
            ### 🎯 Adaboost model(Adaptive Boosting)
            **Type:** Machine Learning Classifier (Ensemble)
            **Algorithm:** Adaboost with Decision Trees
            **Features:** TF-IDF vectors (unigrams + bigrams)
            
            **Strengths:**
            - Combines weak learners into a strong model
            - Often achieves higher accuracy than single model
            - Reduces bias and variancen
            - Performs well on imbalanced datasets
            """)
              
    

            
        with col4:
            st.markdown("""
            ### 🎯 CNN (Convolutional Neural Network)
            **Type:** Deep Learning Classifie
            **Algorithm:** CNN for Text Classification
            **Features:** Word Embeddings (e.g., GloVe), Convolution Filter
            
            **Strengths:**
            - Captures local patterns and n-grams effectively
            - Good for sentence-level text classification

            - Performs well on large datasets
            - Learns hierarchical feature representations
            """)
 
        with col5:
            st.markdown("""
            ### 🎯 LSTM model
            **Type:** Learning Classifier (RNN variant)
            **Algorithm:** LSTM
            **Features:** Word Embeddings, Sequential Modeling
            
            **Strengths:**
            - Excellent for modeling long-term dependencies
            - Ideal for sequential or time-series text data
            - Good performance on text classification
            - Ideal for sequential or time-series text data
            """)
        

        with col6:
            st.markdown("""
            ### 🎯 RNN model
            **Type:** Deep Learning Classifierl
            **Algorithm:** RNN
            **Features:** Features: Word Embeddings, Sequential Modeling
)
            
            **Strengths:**
            - Fast training and prediction
            - Learns temporal dependencies in sequences
            - Good performance on text classification
            - Simpler than LSTM, faster to train (but can suffer from vanishing gradient
            """)


        # Feature engineering info
        st.subheader("🔤 Feature Engineering")
        st.markdown("""
        **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Max Features:** 5,000 most important terms
        - **N-grams:** Unigrams (1-word) and Bigrams (2-word phrases)
        - **Min Document Frequency:** 2 (terms must appear in at least 2 documents)
        - **Stop Words:** English stop words removed
        """)
        
        # File status
        st.subheader("📁 Model Files Status")
        file_status = []
        
        files_to_check = [
            ("svm_model.pkl", "svm Pipeline", models.get('pipeline_available', False)),
            ("tfidf_vectorizer.pkl", "TF-IDF Vectorizer", models.get('vectorizer_available', False)),
            ("adaboost_model.pkl", "adaboost Classifier", models.get('adaboost_available', False)),
            ("decision_tree_model.pkl", "decision tree Classifier", models.get('decision_tree_available', False)),
            ("CNN.pkl", "CNN Classifier", models.get('CNN_available', False)),
            ("LSTM.pkl", "LSTM Classifier", models.get('LSTM_available', False)),
            ("RNN.pkl", "RNN Classifier", models.get('RNN_available', False)),
        ]

#         files_to_check = [
#     ("svm_model.pkl", "svm Pipeline", models.get('pipeline_available', False)),
#     ("tfidf_vectorizer.pkl", "TF-IDF Vectorizer", models.get('vectorizer_available', False)),
#     ("adaboost_model.pkl", "adaboost Classifier", models.get('adaboost_available', False)),
#     ("decision_tree_model.pkl", "decision tree Classifier", models.get('decision_tree_available', False)),  # <- FIXED COMMA
#     ("CNN.pkl", "CNN Classifier", models.get('CNN_available', False)),
#     ("LSTM.pkl", "LSTM Classifier", models.get('LSTM_available', False)),
#     ("RNN.pkl", "RNN Classifier", models.get('RNN_available', False)),
# ]

        
        for filename, description, status in files_to_check:
            file_status.append({
                "File": filename,
                "Description": description,
                "Status": "✅ Loaded" if status else "❌ Not Found"
            })
        
        st.table(pd.DataFrame(file_status))
        
        # Training information
        st.subheader("📚 Training Information")
        st.markdown("""
        **Dataset:** Product Review text Analysis
        - **Classes:** Human and AI text
        - **Preprocessing:** Text cleaning, tokenization, TF-IDF vectorization
        - **Training:** Both models trained on same feature set for fair comparison
        """)
        
    else:
        st.warning("Models not loaded. Please check model files in the 'models/' directory.")

# ============================================================================
# HELP PAGE
# ============================================================================

elif page == "❓ Help":
    st.header("❓ How to Use This App")
    
    with st.expander("🔮 Single Prediction"):
        st.write("""
        1. **Select a model** from the dropdown (Logistic Regression or Multinomial Naive Bayes)
        2. **Enter text** in the text area (product reviews, comments, feedback)
        3. **Click 'Predict'** to get text analysis results
        4. **View results:** prediction, confidence score, and probability breakdown
        5. **Try examples:** Use the provided example texts to test the models
        """)
    
    with st.expander("📁 Batch Processing"):
        st.write("""
        1. **Prepare your file:**
           - **.txt file:** One text per line
           - **.csv file:** Text in the first column
        2. **Upload the file** using the file uploader
        3. **Select a model** for processing
        4. **Click 'Process File'** to analyze all texts
        5. **Download results** as CSV file with predictions and probabilities
        """)
    
    with st.expander("⚖️ Model Comparison"):
        st.write("""
        1. **Enter text** you want to analyze
        2. **Click 'Compare All Models'** to get predictions from both models
        3. **View comparison table** showing predictions and confidence scores
        4. **Analyze agreement:** See if models agree or disagree
        5. **Compare probabilities:** Side-by-side probability charts
        """)
    
    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues and Solutions:**
        
        **Models not loading:**
        - Ensure model files (.pkl) are in the 'models/' directory
        - Check that required files exist:
          - tfidf_vectorizer.pkl (required)
          - svm_model.pkl (for SVM pipeline)
          - adaboost_model.pkl (for adaboost individual)
          - decision_tree_model.pkl (for decision tree model individual)
          - CNN.pkl (for CNN model individual)
          - LSTM.pkl (for LSTM model individual)
          - RNN.pkl (for RNN model individual)
                 
                 
        
        **Prediction errors:**
        - Make sure input text is not empty
        - Try shorter texts if getting memory errors
        - Check that text contains readable characters
        
        **File upload issues:**
        - Ensure file format is .txt or .csv
        - Check file encoding (should be UTF-8)
        - Verify CSV has text in the first column
        """)
    
    # System information
    st.subheader("💻 Your Project Structure")
    st.code("""
    streamlit_ml_app/
    ├── app.py                              # Main application
    ├── requirements.txt                    # Dependencies
    ├── models/                            # Model files
    │   ├── sentiment_analysis_pipeline.pkl # LR complete pipeline
    │   ├── tfidf_vectorizer.pkl           # Feature extraction
    │   ├── logistic_regression_model.pkl  # LR classifier
    │   └── multinomial_nb_model.pkl       # NB classifier
    └── sample_data/                       # Sample files
        ├── sample_texts.txt
        └── sample_data.csv
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.info("""
**ML Text Classification App**
Built with Streamlit

**Models:** 
- 📈 SVM
- Adaboost
- Decision tree
 - CNN : Convolutional Neural Network
 - LSTM: Long Short-Term Memory (a special kind of RNN)
-  RNN: Recurrent Neural Networ


**Framework:** 
 
• Main Framework: Streamlit with Python
• scikit-learn (machine learning)
• pyTorch (Deep Learning)
• pandas/numpy (data processing)
Document Processing
• PyPDF2/pdfplumber (PDF text extraction)
• python-docx (Word document processing)
• NLTK/spaCy (text preprocessing       
                        
**Deployment:** Streamlit Cloud Ready
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit | Machine Learning Text Classification Demo | By Maaz Amjad<br>
    <small>As a part of the courses series **Introduction to Large Language Models/Intro to AI Agents**</small><br>
    <small>This app demonstrates text analysis using trained ML models</small>
</div>
""", unsafe_allow_html=True)



