# config.py

# Data paths
en_path = 'sample\TM-training-set\english.txt'
ch_path = 'sample\TM-training-set\chinese.txt'

# Model parameters
model_dim = 256
num_heads = 8
num_encoder_layers = 2
num_decoder_layers = 2
ffn_dim = 512

# Training settings
batch_size = 40
num_epochs = 100
label_smoothing = 0.1
warmup_steps = 2000

# Optimization
adam_betas = (0.9, 0.98)
adam_eps = 1e-9

# Output paths
model_save_dir = 'model'
loss_log_path = 'loss.txt'

# Evaluation
bleu_eval_sample_size = 100
