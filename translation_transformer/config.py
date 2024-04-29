from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "lr": 10**-4,
        "epochs": 25,
        "seq_len": 750,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "es",
        "model_folder": "weights",
        "model_file": "mini_transformer_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "exp_name": "runs/mini_transformer",
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_file = config['model_file']
    model_filename = f'{model_file}{epoch}.pt'
    return str(Path('.')/model_folder/model_filename)

