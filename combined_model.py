import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
def load_data(file_path='data_backup.csv'):
    df = pd.read_csv(file_path, sep=';')
    return df

def prepare_data(df):
    # Define key features
    key_features = [
        'Scholarship holder',
        'Course',
        'Nacionality',
        'International',
        "Mother's qualification",
        "Father's qualification",
        'Displaced',
        'Debtor',
        'Curricular units 1st sem (grade)',
        'Curricular units 2nd sem (grade)'
    ]
    
    # Create a copy for preparation
    df_prep = df.copy()
    
    # Handle missing values
    df_prep = df_prep.fillna(df_prep.mean())
    
    # Feature engineering
    df_prep['Grade_Progression'] = df_prep['Curricular units 2nd sem (grade)'] - df_prep['Curricular units 1st sem (grade)']
    df_prep['Family_Education'] = (df_prep["Mother's qualification"] + df_prep["Father's qualification"]) / 2
    
    # Feature scaling
    scaler = StandardScaler()
    numeric_features = ['Curricular units 1st sem (grade)', 
                       'Curricular units 2nd sem (grade)',
                       'Grade_Progression',
                       'Family_Education']
    df_prep[numeric_features] = scaler.fit_transform(df_prep[numeric_features])
    
    return df_prep, key_features

def train_combined_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use Decision Tree for feature selection
    dt_selector = DecisionTreeClassifier(random_state=42)
    dt_selector.fit(X_train, y_train)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': dt_selector.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Select features based on importance threshold
    selector = SelectFromModel(dt_selector, prefit=True, threshold='mean')
    X_selected = selector.transform(X)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Train Naive Bayes with selected features
    nb = GaussianNB()
    nb.fit(X_train_selected, y_train)
    
    return nb, selector, feature_importance, X_test_selected, y_test

def evaluate_model(model, X_test, y_test, feature_importance):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Print feature importance
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance using Decision Tree')
    plt.tight_layout()
    plt.show()

def main():
    # Load and prepare data
    df = load_data()
    df_prep, key_features = prepare_data(df)
    
    # Prepare features and target
    X = df_prep[key_features]
    y = df_prep['Target']
    
    # Train combined model
    nb_model, selector, feature_importance, X_test_selected, y_test = train_combined_model(X, y)
    
    # Evaluate model
    evaluate_model(nb_model, X_test_selected, y_test, feature_importance)
    
    # Print selected features
    selected_features = X.columns[selector.get_support()].tolist()
    print("\nSelected Features:")
    for feature in selected_features:
        importance = feature_importance[feature_importance['Feature'] == feature]['Importance'].values[0]
        print(f"- {feature}: {importance:.3f}")

if __name__ == "__main__":
    main() 