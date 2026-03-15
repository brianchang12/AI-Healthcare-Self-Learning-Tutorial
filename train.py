from pathlib import Path
import pandas as pd
import argparse
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Classifier(Enum):
    ada_boost = 1
    extra_trees = 2
    random_forest = 3
    bagging = 4

def parse_args():
    parser = argparse.ArgumentParser(description="Train a malignant/benign tumor diagnoses classifier using analysis on radiology notes.")
    parser.add_argument(
        "--model",
        type=str,
        choices=[c.name for c in Classifier],
        default=Classifier.ada_boost,
        help="Classifier to use"
    )
    return parser.parse_args()

def classifier_scores(truth, prediction):

    accuracy_val = accuracy_score(truth, prediction)
    precision_val = precision_score(truth, prediction)
    recall_val = recall_score(truth, prediction)
    f1_val = f1_score(truth, prediction)

    return (precision_val, recall_val, accuracy_val, f1_val)

if __name__ == "__main__":

    args = parse_args()

    train_features_path = Path("./data/train/features.csv")
    train_results_path = Path("./data/train/results.csv")
    # Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Read csv to dataframe)
    train_features_df = pd.read_csv(train_features_path)
    train_results_df = pd.read_csv(train_results_path)
    # End of: Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Read csv to dataframe)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    transformed_train_features_df = vectorizer.fit_transform(train_features_df["notes"])
    
    if Classifier.ada_boost.name == args.model: 
        classifier = AdaBoostClassifier(random_state=42)
    elif Classifier.extra_trees.name == args.model:
        classifier = ExtraTreesClassifier(random_state=42)
    elif Classifier.random_forest.name == args.model:
        classifier = RandomForestClassifier(random_state=42)
    else:
        classifier = BaggingClassifier(random_state=42)

    classifier = classifier.fit(transformed_train_features_df, train_results_df["is_malignant_diagnoses"])

    train_prediction = classifier.predict(transformed_train_features_df)

    validation_features_path = Path("./data/validation/features.csv")
    validation_results_path = Path("./data/validation/results.csv")

     # Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Read csv to dataframe)
    validation_features_df = pd.read_csv(validation_features_path)
    validation_results_df = pd.read_csv(validation_results_path)
    # End of: Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Read csv to dataframe)

    transformed_validation_features_df = vectorizer.transform(validation_features_df["notes"])

    validation_prediction = classifier.predict(transformed_validation_features_df)

    train_scores = classifier_scores(train_results_df["is_malignant_diagnoses"], train_prediction)
    validation_scores = classifier_scores(validation_results_df["is_malignant_diagnoses"], validation_prediction)

    classifier_name = "AdaBoostClassifier"

    train_output = f"""Training Scores - {classifier_name}:

Precision: {train_scores[0]:.5f}, Recall: {train_scores[1]:.5f}, Accuracy: {train_scores[2]:.5f}, F1: {train_scores[3]:.5f}
    """
    
    validation_output = f"""Validation Scores - {classifier_name}:

Precision: {validation_scores[0]:.5f}, Recall: {validation_scores[1]:.5f}, Accuracy: {validation_scores[2]:.5f}, F1: {validation_scores[3]:.5f}
    """
    

    print("==============================================================")
    print(train_output)
    print(validation_output)
    print("==============================================================")



