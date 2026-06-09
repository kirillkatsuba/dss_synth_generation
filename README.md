# Synthetic Tabular Data Generation for Data Storage Systems

## CTGAN
Training procedure:
```[python]
python3 ctgan_generator.py {dataset_name}.csv \
  --output-model /home/kkatsuba/work_flow/dss_generation/trained_models/ctgan/ctgan_model_{dataset_name}.pkl \
  --gpu \
  --verbose \
  --output-data /home/kkatsuba/work_flow/dss_generation/synth_data/ctgan/ctgan_synth_{dataset_name}.csv
```

Sampling:
```[python]
python3 ctgan_generator.py data.csv \
  --sample \
  --load-model /home/kkatsuba/work_flow/dss_generation/trained_models/ctgan_data.pkl \
  --output-data /home/kkatsuba/work_flow/dss_generation/synth_data/synth_data.csv
```
