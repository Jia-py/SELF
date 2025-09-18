# SELF

Our code is based on the feature selection framework [ERASE](https://github.com/Applied-Machine-Learning-Lab/ERASE), please put the `selfs.py` under `models/fs` path

## Datasets

You can download the datasets from [movielens](https://github.com/datawhalechina/torch-rechub/tree/main/examples/matching), [aliccp](https://github.com/datawhalechina/torch-rechub/tree/main/examples/ranking), [kuairand](https://kuairand.com/)

## Running

1. Feature Importance Extraction stage

```bash
python predict.py --model=gpt-4-turbo --dataset=movielens
```

2. Feature Importance Refinement stage

```bash
python fs_run.py --dataset=movielens-1m --fs=selfs --retrain=false
```

3. Retraining stage

```bash
python llm_selection.py
python fs_run.py --dataset=movielens-1m --model=dcn --train_or_search=false --rank_path=selfs_movielens-1m --k={k}
```
