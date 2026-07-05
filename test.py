from sklearn.model_selection import (cross_val_score,GridSearchCV)
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df  = pd.read_csv("data/driver_churn.csv")
model = RandomForestClassifier(
        random_state=42
    )
X = df[[
    "loads_completed","last_login_days","rating"
]]
y=df["churn"]
scores = cross_val_score(
    model,X,y,cv=5
)
param_grid = {
    "n_estimators":[10,50,100,200],
    'max_depth':[2,4,6,None],
    "min_samples_split":[2,4,8]
}

grid=GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy")
grid.fit(X,y)

print(scores)
print(scores.mean())
print(grid.best_params_)
print(grid.best_score_)
