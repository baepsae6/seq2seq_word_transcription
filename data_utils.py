import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from vocab import Vocab
from collections import Counter
from IPython.display import clear_output
import matplotlib.pyplot as plt


class TranscriptionsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.t_vocab = pickle.load(open('transcription_vocab.pickle', 'rb'))
        self.w_vocab = pickle.load(open('word_vocab.pickle', 'rb'))
        self.word = self.data['Word'].values
        self.transcription = self.data['Transcription'].values
    
    def __getitem__(self, idx):
        word = self.word[idx]
        transcription = self.transcription[idx]
        word  = self.w_vocab.sent2idx(word)
        transcription = self.t_vocab.sent2idx(transcription)
        sample = {'word': word, 'transcription': transcription}
        return sample
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, dicts): 
        pad_token = self.t_vocab.token2idx('<pad>')
        sos_token = self.t_vocab.token2idx('<sos>')
        eos_token = self.t_vocab.token2idx('<eos>')
        words_padded = []
        corpus_size = len(dicts)
        len_words_list = [len(d['word']) for d in dicts]
        words_list = [d['word'] for d in dicts]
        
        len_transcriptions_list = [len(d['transcription']) for d in dicts]
        transcriptions = [i['transcription'] for i in dicts]
       
        sorted_len_words, sorted_words, sorted_len_transcriptions, sorted_transcriptions = list(zip(*sorted(zip(
            len_words_list, words_list, len_transcriptions_list, transcriptions),key=lambda x: x[0] ,reverse=True)))      
        max_lens_word = max(sorted_len_words)
        max_lens_transcription = max(sorted_len_transcriptions)

        
        words_padded = [sorted_words[i] + [pad_token] * (max_lens_word - sorted_len_words[i]) for i in range(corpus_size)]
        sos_transcriptions_padded = [[sos_token] + sorted_transcriptions[i] + [pad_token] * (max_lens_transcription - sorted_len_transcriptions[i]) for i in range(corpus_size)]
        eos_transcriptions_padded = [sorted_transcriptions[i] + [eos_token] + [pad_token] * (max_lens_transcription - sorted_len_transcriptions[i]) for i in range(corpus_size)]
        words_padded = torch.LongTensor(words_padded)
        
        sos_transcriptions_padded = torch.LongTensor(sos_transcriptions_padded)
        eos_transcriptions_padded = torch.LongTensor(eos_transcriptions_padded)
        
        return (words_padded, sos_transcriptions_padded, eos_transcriptions_padded)


def tokenize(word):
    return [char for char in word if char!=' ']


def tokenize_data():
    data  = pd.read_csv('data/transcriptions/train.csv')
    data['Word'] = data['Word'].apply(lambda row: tokenize(row) if type(row)!= float else 0)
    data = data[(data['Word']!=0)]
    data['Transcription'] = data['Transcription'].apply(lambda row: tokenize(row))
    return data


def create_vocab(data):
    data = tokenize_data()
    # Creates vocab for words and saves it in a pickle file
    word_list1 = []
    for word in tqdm(data['Word'].values):
        word_list1 += word
        word_counter = Counter(word_list1)
    vocab1 = Vocab(word_counter)
    with open('word_vocab.pickle', 'wb') as f:
        pickle.dump(vocab1, f)
    # Creates vocab for transcriptions and saves it in a pickle file    
    word_list2 = []
    for word in tqdm(data['Transcription'].values):
        word_list2 += word
        word_counter = Counter(word_list2)
    vocab2 = Vocab(word_counter)
    with open('transcription_vocab.pickle', 'wb') as f:
        pickle.dump(vocab2, f)
        
        
def load_data():
    data = tokenize_data()
    
    X_train, X_test = train_test_split(data, test_size=0.33, random_state=42)
    
    train_dataset = TranscriptionsDataset(X_train)
    test_dataset = TranscriptionsDataset(X_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32,
                                  shuffle=True, collate_fn=train_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32,
                                 shuffle=False, collate_fn=test_dataset.collate_fn)
    return train_dataloader, test_dataloader


def plot(epoch, step, train_losses):
    clear_output()
    plt.title(f'Epochs {epoch}, step {step}')
    plt.plot(train_losses)
    plt.show()

