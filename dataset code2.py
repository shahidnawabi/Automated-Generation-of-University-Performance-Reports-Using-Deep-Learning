import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().splitlines()
    return pd.DataFrame(text, columns=['text'])

def write_text_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in data:
            file.write(line + '\n')

def generate_synthetic_text(prompt, max_length=50, num_samples=100):
    synthetic_texts = []
    for _ in range(num_samples):
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        synthetic_texts.append(generated_text)
    return synthetic_texts

def generate_synthetic_data(data, method='gpt2', num_samples=100):
    if method == 'gpt2':
        prompt = " ".join(data.sample(1).values[0])
        synthetic_texts = generate_synthetic_text(prompt, num_samples=num_samples)
        return pd.DataFrame(synthetic_texts, columns=['text'])
    else:
        raise ValueError("Unknown method")

def main():
    # Load your real-world report data
    real_data = read_text_file('real_text_data.txt')
    
    # Split the data
    train_data, test_data = train_test_split(real_data, test_size=0.2, random_state=42)
    
    # Generate synthetic text data
    synthetic_data_gpt2 = generate_synthetic_data(train_data, method='gpt2', num_samples=1000)
    
    # Combine real and synthetic data
    combined_data_A1 = pd.concat([train_data, synthetic_data_gpt2]).reset_index(drop=True)
    
    # Save the datasets
    write_text_file('combined_data_A1.txt', combined_data_A1['text'].tolist())
    write_text_file('synthetic_data_gpt2.txt', synthetic_data_gpt2['text'].tolist())

if __name__ == '__main__':
    main()
