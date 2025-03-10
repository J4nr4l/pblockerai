import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, AdamW, GPT2Tokenizer
from utils import load_data, preprocess_data, save_model

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = load_data(file_path)
        self.data = preprocess_data(self.data)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        encoding = self.tokenizer(item['text'], return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        encoding['labels'] = encoding['input_ids']
        return encoding

def train_model(model, dataloader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].squeeze(1).to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

def main():
    with open('configs/training_args.json') as f:
        training_args = json.load(f)

    with open('configs/model_config.json') as f:
        model_config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = training_args.get('model_name', 'gpt2')
    model = GPT2LMHeadModel.from_pretrained(model_name, **model_config).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    optimizer = AdamW(model.parameters(), lr=training_args['learning_rate'], weight_decay=training_args['weight_decay'])

    train_dataset = TextDataset('data/train.csv', tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=training_args['batch_size'], shuffle=True)

    train_model(model, train_dataloader, optimizer, device, training_args['num_epochs'])

    save_model(model, tokenizer, './model')

if __name__ == '__main__':
    main()