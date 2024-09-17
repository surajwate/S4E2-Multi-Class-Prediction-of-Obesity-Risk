import pandas as pd
from sklearn.model_selection import StratifiedKFold

def create_fold(fold):
    # import data
    data = pd.read_csv('./input/train.csv')

    data['kfold'] = -1

    data = data.sample(frac=1).reset_index(drop=True)

    y = data.NObeyesdad.values

    skf = StratifiedKFold(n_splits=fold)

    for f, (t_, v_) in enumerate(skf.split(X=data, y=y)):
        data.loc[v_, 'kfold'] = f

    data.to_csv('./input/train_folds.csv', index=False)

if __name__ == '__main__':
    create_fold(5)