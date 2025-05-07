import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# Features from the decision tree
TREE_FEATURES = [
    'Curricular units 2nd sem (approved)',
    'Course',
    'Tuition fees up to date',
    'Curricular units 2nd sem (grade)'
]

def load_data(file_path='data_backup.csv'):
    df = pd.read_csv(file_path, sep=';')
    return df

def prepare_data(df):
    # Only keep the relevant features and the target
    features = TREE_FEATURES
    df_prep = df.copy()
    df_prep = df_prep.fillna(df_prep.mean())
    X = df_prep[features]
    y = df_prep['Target']
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def main():
    df = load_data()
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {acc:.3f}")

    # ROC Curve and AUC
    if len(set(y_test)) == 2:  # Only plot if binary classification
        y_proba = nb.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc:.3f}")
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Naive Bayes')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("ROC curve is only available for binary classification.")

if __name__ == "__main__":
    main() 