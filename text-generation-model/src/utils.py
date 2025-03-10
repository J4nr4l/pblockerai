import pandas as pd
import torch

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    
    return data

def evaluate_model(model, tokenizer, test_data):
   
    pass

def save_model(model, tokenizer, file_path):
    model.save_pretrained(file_path)
    tokenizer.save_pretrained(file_path)

def load_model(file_path):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    model = GPT2LMHeadModel.from_pretrained(file_path)
    tokenizer = GPT2Tokenizer.from_pretrained(file_path)
    return model, tokenizer