import joblib

# Load the preprocessed data from the .pkl file
X_train, X_test, y_train, y_test = joblib.load('preprocessed_data.pkl')

# Display the shapes of the loaded data
print("Shapes of the datasets:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# If you want to inspect some sample data
print("\nSample data from X_train:")
print(X_train[:5])

print("\nSample labels from y_train:")
print(y_train[:5])
