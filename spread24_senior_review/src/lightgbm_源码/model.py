import lightgbm as lgb

def fit_predict(train_x, train_y, test_x, cfg):
    p=cfg["model"]
    model=lgb.LGBMRegressor(objective=p["objective"],n_estimators=p["n_estimators"],learning_rate=p["learning_rate"],num_leaves=p["num_leaves"],min_child_samples=p["min_child_samples"],subsample=p["subsample"],colsample_bytree=p["colsample_bytree"],reg_lambda=p["reg_lambda"],random_state=cfg["seed"],verbosity=-1,n_jobs=-1)
    model.fit(train_x,train_y)
    return model.predict(test_x)

