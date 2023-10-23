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