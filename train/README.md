# Training models

Install python packages
```
pip insntall -r requirements.txt
```

Have a pgn of training data (example sources, lichess history, ccrl games)

Feature Extraction
```
../target/release/princhess -t /path/to/pgn
```

Split training data

```
split -l 1000000 train_data.libsvm train_data.
```

Train data

```
python training.py
```


