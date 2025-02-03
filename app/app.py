import streamlit as st
import torch
import pickle
from models.classes import *
from models.utils import get_text_transform, mmtokenizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model metadata
meta = pickle.load(open("./models/param-multiplicative.pkl", "rb"))

# Define token and vocabulary transformations
token_transform = meta["token_transform"]
vocab_transform = meta["vocab_transform"]
text_transform = get_text_transform(token_transform, vocab_transform)

# Language definitions
SRC_LANGUAGE = "en"
TRG_LANGUAGE = "my"

# Model parameters
input_dim = len(vocab_transform[SRC_LANGUAGE])
output_dim = len(vocab_transform[TRG_LANGUAGE])
hid_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1

# Special tokens
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SRC_PAD_IDX = PAD_IDX
TRG_PAD_IDX = PAD_IDX

# Load encoder and decoder
enc = Encoder(input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device)
dec = Decoder(output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, device)

# Load Seq2Seq model
model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.load_state_dict(torch.load("models/Seq2SeqTransformer-multiplicative.pt", map_location=device))
model.eval()

# def translate_text(prompt):
#     """Translates an English sentence into Myanmar."""
#     max_seq = 100
    
#     src_text = text_transform[SRC_LANGUAGE](prompt).to(device)
#     src_text = src_text.reshape(1, -1)
#     src_mask = model.make_src_mask(src_text)
    
#     with torch.no_grad():
#         enc_output = model.encoder(src_text, src_mask)
    
#     outputs = []
#     input_tokens = [EOS_IDX]
    
#     for _ in range(max_seq):
#         with torch.no_grad():
#             starting_token = torch.LongTensor(input_tokens).unsqueeze(0).to(device)
#             trg_mask = model.make_trg_mask(starting_token)
#             output, _ = model.decoder(starting_token, enc_output, trg_mask, src_mask)
        
#         pred_token = output.argmax(2)[:, -1].item()
#         input_tokens.append(pred_token)
#         outputs.append(pred_token)
        
#         if pred_token == EOS_IDX:
#             break
    
#     trg_tokens = [vocab_transform[TRG_LANGUAGE].get_itos()[i] for i in outputs]
#     return " ".join(trg_tokens[1:-1])

def translate_text(prompt):
    """Translates an English sentence into Myanmar."""
    max_seq = 100

    print(f"Received input: {prompt}")

    src_text = text_transform[SRC_LANGUAGE](prompt).to(device)
    src_text = src_text.reshape(1, -1)
    src_mask = model.make_src_mask(src_text)

    print(f"Tokenized input: {src_text}")

    with torch.no_grad():
        enc_output = model.encoder(src_text, src_mask)
    
    outputs = []
    input_tokens = [EOS_IDX]

    for i in range(max_seq):
        with torch.no_grad():
            starting_token = torch.LongTensor(input_tokens).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(starting_token)
            output, _ = model.decoder(starting_token, enc_output, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        input_tokens.append(pred_token)
        outputs.append(pred_token)
        
        print(f"Step {i}: Predicted token {pred_token}")

        if pred_token == EOS_IDX:
            break
    
    print(f"Final token sequence: {outputs}")

    trg_tokens = [vocab_transform[TRG_LANGUAGE].get_itos()[i] for i in outputs]
    print(f"Translated text tokens: {trg_tokens}")

    return " ".join(trg_tokens[1:-1])


# Streamlit UI
st.title("English to Myanmar Translator")
st.write("Enter a sentence in English, and get the Myanmar translation.")

user_input = st.text_input("Enter text in English:", "Hello, how are you?")
if st.button("Translate"):
    translation = translate_text(user_input)
    st.write("### Translated Text:")
    st.success(translation)
