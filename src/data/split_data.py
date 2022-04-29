from sklearn.model_selection import train_test_split
from src.features.build_features import df

# # split the data into 80 to 20
# end = int(0.8 * df.shape[0])
# training_set = df.iloc[:end]
# validation_set = df.iloc[end+1:]
# #Prepare data for train_test_split
# X_validation = validation_set.drop(['damage_grade'], axis=1)
# y_validation = validation_set['damage_grade']

# TODO temporary measures for smaller dataset and XGBoost compatibility
# df['damage_grade'] = df['damage_grade'] - 1
# df = df.head(1000)

# Prepare data for train_test_split
y = df['damage_grade'].to_numpy()
X = df.copy()
X.drop(['damage_grade'], axis=1, inplace=True)
X = X.to_numpy()

# train test split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)

