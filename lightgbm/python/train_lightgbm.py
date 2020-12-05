import os
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings("ignore")

par_path = "/lightgbm"


def process_data():
    rating = pd.read_csv(os.path.join(par_path, "rating.csv"))
    rating = rating[rating.rating != -1]
    anime = pd.read_csv(os.path.join(par_path, "anime.csv"))
    anime = anime.rename(columns={"rating": "web_rating"}).drop(["name", "genre"], axis=1)
    data = rating.merge(anime, how="inner", on="anime_id")
    label = data["rating"]
    data = data.drop(["rating"], axis=1)
    data[data.episodes == "Unknown"] = 0
    data["episodes"] = data["episodes"].astype(int)
    encoder = OrdinalEncoder()
    data["user_id"] = encoder.fit_transform(data[["user_id"]]).astype(int)
    data["anime_id"] = encoder.fit_transform(data[["anime_id"]]).astype(int)
    data.loc[data["type"].notnull(), "type"] = encoder.fit_transform(
        data.loc[data["type"].notnull(), ["type"]].astype(str)
    ).astype(int)
    return data, label


if __name__ == "__main__":
    data, label = process_data()
    print(data.shape)
    print(data.head())

    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "metric": "rmse",
        "num_iterations": 100,
        "learning_rate": 1.0,
        "num_leaves": 77,
        "num_threads": 4,
        "min_data_in_leaf": 5,
        "max_depth": 0,
        "min_sum_hessian_in_leaf": 0.0,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "feature_fraction": 0.7,
        "early_stopping_round": 5,
        "first_metric_only": False,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "min_gain_to_split": 0.0,
        "max_cat_threshold": 32,
        "verbosity": 0,
        "max_bin": 255,
        "use_missing": True,
        "zero_as_missing": False,
        "header": True,
        "is_unbalance": False,
        "seed": 0,
    }

    X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.2)
    train_data = lgb.Dataset(X_train, y_train, categorical_feature=["user_id", "anime_id", "type"])
    eval_data = lgb.Dataset(X_val, y_val, categorical_feature=["user_id", "anime_id", "type"])
    model = lgb.train(params, train_data, valid_sets=[eval_data])
    model.save_model('model_lgb.txt', num_iteration=model.best_iteration)
