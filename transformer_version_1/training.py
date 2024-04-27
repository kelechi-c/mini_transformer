import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

from dataset_prep import BilingualDataset, causal_mask
from mini_transformer import build_transformer


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
           
def create_tokenizer(config, ds, lang):
    tokenizer_path = Path(config('tokenizer_file').format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer

def get_dataset(config):
    ds_raw = load_dataset('opus_books', f'{config['lang_src']}+{config['lang_tgt']}', split='train')
    
    tokenizer_src = create_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = create_tokenizer(config, ds_raw, config['lang_tgt'])
    
    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_data, val_data = random_split(ds_raw, [train_size, val_size])
    
    train_dataset = BilingualDataset(train_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    valid_dataset = BilingualDataset(val_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    
    return train_loader, val_loader, tokenizer_src, tokenizer_tgt, 


def build_model(config, vocab_src_len, vocab_tgt_len):
    mini_transformer = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    
    return mini_transformer
    
