import pandas as pd

train = pd.read_csv("train.csv")
train.head()

test = pd.read_csv("test.csv")
test.head()

print("Train shape: " , train.shape)
print("Test shape: ",test.shape)

print(train.dtypes)

X = train[['fire_location_latitude', 'fire_location_longitude', 'fire_origin', 'true_cause', 'fire_type', 'weather_conditions_over_fire', 'fuel_type']]
X.head()

y = train['size_class']
y.unique()

null_value_stats = X.isnull().sum(axis=0)
null_value_stats[null_value_stats != 0]

X = X.fillna('Unknown') 

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.75, random_state=42)

import category_encoders as ce

ohe = ce.OneHotEncoder(handle_unknown='value', use_cat_names=True)
X_train_ohe = ohe.fit_transform(X_train)
X_train_ohe.sample(5)

X_valid_ohe = ohe.transform(X_valid)
X_valid_ohe.sample(5)