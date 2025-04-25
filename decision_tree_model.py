import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Read the data
print("Loading data...")
df = pd.read_csv('data_backup.csv', sep=';')

# Select features for the model
features = [
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)'
]

# Prepare the data
print("Preparing data...")
X = df[features].copy()
y = df['Target']

# Handle missing values
X = X.fillna(X.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree
dt = DecisionTreeClassifier(
    max_depth=5,  # Limit tree depth to prevent overfitting
    min_samples_split=50,  # Minimum samples required to split a node
    min_samples_leaf=25,  # Minimum samples required in a leaf node
    random_state=42
)

# Fit the model
print("Training Decision Tree model...")
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create confusion matrix visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt, 
          feature_names=features,
          class_names=['Graduate', 'Dropout'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree for Student Dropout Prediction')
plt.show()

# Calculate and plot feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': dt.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature')
plt.title('Feature Importance in Decision Tree')
plt.tight_layout()
plt.show()

# Print decision tree rules
print("\nDecision Tree Rules:")
print(export_text(dt, feature_names=features))

# Function to analyze decision paths
def analyze_path(X, tree, feature_names):
    """Analyze the path of a sample through the decision tree"""
    path = tree.decision_path(X)
    node_indices = path.indices
    
    print("\nPath Analysis:")
    for node_idx in node_indices:
        if tree.tree_.children_left[node_idx] != -1:  # If not a leaf node
            feature = feature_names[tree.tree_.feature[node_idx]]
            threshold = tree.tree_.threshold[node_idx]
            print(f"Node {node_idx}: {feature} <= {threshold:.2f}")

# Analyze sample paths
print("\nSample Path Analysis (for first 3 test cases):")
for i in range(3):
    print(f"\nSample {i+1}:")
    print(f"Actual class: {y_test.iloc[i]}")
    print(f"Predicted class: {y_pred[i]}")
    analyze_path(X_test[i:i+1], dt, features)

# Calculate and print accuracy for each class
class_accuracy = {}
for class_name in dt.classes_:
    class_mask = y_test == class_name
    class_accuracy[class_name] = np.mean(y_pred[class_mask] == y_test[class_mask])

print("\nAccuracy by Class:")
for class_name, accuracy in class_accuracy.items():
    print(f"{class_name}: {accuracy:.2%}")

# Save the model predictions
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results_df.to_csv('decision_tree_predictions.csv', index=False)
print("\nResults have been saved to 'decision_tree_predictions.csv'") 