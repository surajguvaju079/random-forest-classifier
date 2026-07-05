import joblib
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#load dataset 
df = pd.read_csv("data/driver_churn.csv")


X=df[[
    "loads_completed",
    "last_login_days",
    "rating"
]]

y=df["churn"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y);


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train);

joblib.dump(model,"models/churn_model.pkl")
print("Model saved successfully")

