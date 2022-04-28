from sklearn.model_selection import train_test_split
from src.features.build_features import df

# split the data into 80 to 20
end = int(0.8 * df.shape[0])
training_set = df.iloc[:end]
validation_set = df.iloc[end+1:]

#Prepare data for train_test_split
X = training_set.drop(['damage_grade'], axis=1)
y = training_set['damage_grade']

# train,test split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
