python3 ./scripts/ctgan_generator.py dataset/pools/train_hdd_sequential.csv \
  --output-model trained_models/ctgan/ctgan_model_hdd_seq.pkl \
  --gpu \
  --verbose \
  --output-data synth_data/ctgan/ctgan_synth_hdd_seq.csv