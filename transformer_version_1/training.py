import torch
import warnings
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
from config import get_config, get_weights_file_path
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from tqdm.auto import tqdm
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
    ds_raw = load_dataset('opus_books', config['lang_src'] + '-' + config['lang_tgt'], split='train')
    
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
    
    return train_loader, val_loader, tokenizer_src, tokenizer_tgt 


def build_model(config, vocab_src_len, vocab_tgt_len):
    mini_transformer = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    
    return mini_transformer


def training_loop(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = build_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Tensorboard
    writer = SummaryWriter(config['exp_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    initial_epoch = 0
    step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Loading weights from {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    for epoch in tqdm(range(initial_epoch, config['epochs'])):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            
            encoder_output = model.encoder(encoder_input, encoder_mask)
            decoder_output = model.decoder(encoder_output, encoder_mask, decoder_input, decoder_mask)
            projection_output = model.projection(decoder_output)
            
            label = batch['label'].to(device)
            loss = loss_fn(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f'loss': f'{loss.item():.2f}'})
            
            writer.add_scalar('loss', loss.item(), step)
            writer.flush()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'step': step,
                'optimizer': optimizer.state_dict()
            }, model_filename)
            
            
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    training_loop(config)