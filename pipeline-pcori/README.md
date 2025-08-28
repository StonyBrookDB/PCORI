# examples for run

# model logreg:
python train.py \
  --dataset ./data/synth \
  --spec ./data/synth/FeatureSpec.json \
  --model logreg


# model mlp:
python train.py --dataset ./data/synth --spec ./data/synth/FeatureSpec.json --model mlp


# model lstm:
python train.py \
  --dataset ./data/synth \
  --spec ./data/synth/FeatureSpec.json \
  --model lstm \
  --epochs 8 \
  --lr 1e-3 \
  --batch_size 128 \
  --hidden_size 128 \
  --num_layers 1 \
  --dropout 0.1 \
  --device cuda

# traditional baselines
python train.py --dataset ./data/synth --spec ./data/synth/FeatureSpec.json --model rf
python train.py --dataset ./data/synth --spec ./data/synth/FeatureSpec.json --model dt

# Bilstm
python train.py --dataset ./data/synth --spec ./data/synth/FeatureSpec.json --model bilstm \
  --epochs 8 --hidden_size 128 --batch_size 128

python train.py --dataset ./data/synth --spec ./data/synth/FeatureSpec.json --model lstm_attn \
  --epochs 8 --hidden_size 128 --batch_size 128

# Graph-only(Patient pooling: mean)
python train.py --dataset ./data/synth --spec ./data/synth/FeatureSpec.json \
  --model gcn --hidden_size 128 --epochs 8 --batch_size 256 --pooling mean --normalize_rows

# Fusion：LSTM + Graph Enc embedding (composed of (T,H) by t_index)
python train.py --dataset ./data/synth --spec ./data/synth/FeatureSpec.json \
  --model lstm_gcn --hidden_size 128 --epochs 8 --batch_size 128 --num_layers 1 --dropout 0.1


# HeteroRGCN（Graph-only）
python train.py --dataset ./data/synth --spec ./data/synth/FeatureSpec.json \
  --model hetero_rgcn --hidden_size 128 --epochs 8 --batch_size 256 --pooling mean --normalize_rows --dropout 0.1

# LIGHTED
python train.py --dataset ./data/synth --spec ./data/synth/FeatureSpec.json \
  --model lighted --hidden_size 128 --epochs 8 --batch_size 128 --num_layers 1 --dropout 0.1 --bidirectional




list eval-run result:
python train.py --dataset ./data/synth --spec ./data/synth/FeatureSpec.json --model lstm \
  --epochs 6 --hidden_size 128 --batch_size 128
# record runs/<timestamp>

eval:
python eval.py --run ./runs/<timestamp> --split test
python eval.py --run ./runs/2025-08-14_07-00-30 --split test

sum eval：
python tools/collect_runs.py --runs_root ./runs

