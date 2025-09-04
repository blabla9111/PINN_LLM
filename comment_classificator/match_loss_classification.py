from transformers import BertTokenizer
import numpy as np
import torch
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 15)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.sigmoid(linear_output)

        return final_layer



def predict_text(model, tokenizer, text, device=None, top_k=2):
    """
    Predict top-k classes for input text

    Args:
        model: Trained model
        tokenizer: Tokenizer for the model
        text: Input text to classify
        device: Device to use (cuda/cpu)
        top_k: Number of top predictions to return

    Returns:
        Tuple of (top class indices, top class probabilities, all probabilities)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare input
    inputs = tokenizer(text,
                       padding='max_length',
                       max_length=128,
                       truncation=True,
                       return_tensors="pt")

    # Move to device
    input_ids = inputs['input_ids'].squeeze(1).to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs) if outputs.min(
        ) < 0 else outputs  # Apply sigmoid if logits

    # Convert to numpy
    probs = probs.cpu().numpy().flatten()

    # Get top-k predictions
    top_indices = np.argsort(probs)[-top_k:][::-1]  # Sort descending
    top_probs = probs[top_indices]

    return top_indices, top_probs, probs

def load_model():
    model = BertClassifier()
    state_dict = torch.load(
    'comment_classificator\saved_models\model_weights_MATCH_LOSS_2_82_0.pth', weights_only=True)  # путь относительно корневой папки
    model.load_state_dict(state_dict)

    return model

def load_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    return tokenizer


labels_level_main = {0: '1',
                     1: '1',
                     2: '2',
                     3: '3',
                     4: '4',
                     5: '2',
                     6: '1',
                     7: '2',
                     8: '3',
                     9: '3',
                     10: '1',
                     11: '2',
                     12: '4',
                     13: '1',
                     14: '2'
                     }

def predict_class_and_sub_class(text):
    # text = "The effect of quarantine is not reflected after day"
    model = load_model()
    tokenizer = load_tokenizer()
    top_indices, top_probs, all_probs = predict_text(model, tokenizer, text)
    print(top_indices)
    for i in range(len(top_indices)):
        top_indices[i] = labels_level_main[top_indices[i]]
    return top_indices, top_probs


# print(predict_class_and_sub_class(
#     "The effect of quarantine is not reflected after day"))


# print("Top predicted classes:")
# for idx, prob in zip(top_indices, top_probs):
#     print(f"Class {idx}: {prob:.4f}")
