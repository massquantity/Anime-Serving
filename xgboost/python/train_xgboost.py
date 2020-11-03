from pprint import pprint
import xgboost as xgb


if __name__ == "__main__":
    train = xgb.DMatrix("../train_data.csv?format=csv&label_column=0")
    test = xgb.DMatrix("../test_data.csv?format=csv&label_column=0")

    param = [
        ('max_depth', 3),
        ('objective', 'reg:squarederror'),
        ('eval_metric', 'rmse'),
        ('eval_metric', 'rmsle'),
        ('eta', 0.2),
        ('min_child_weight', 1)
    ]
    num_round = 5
    watchlist = [(train, 'train'), (test, 'eval')]
    eval_results = {}

    bst = xgb.train(param, train, num_round, watchlist, evals_result=eval_results)
    pprint(eval_results)
