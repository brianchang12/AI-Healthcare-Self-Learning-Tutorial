"""
This is the training script for the classifiers.
"""
# Referenced/Used material from AI healthcare class
# Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (General help, Help with parse_args)
# Referenced/Used previous work completed by me for AI healthcare class
# Referenced/Used: https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
# Referenced/Used: https://codezup.com/natural-language-processing-sentiment-analysis-scikit-learn/
# Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
# Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
# Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
# Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Referenced/Used: https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/
# Referenced/Used: https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/
from pathlib import Path
import pandas as pd
import argparse
from enum import Enum
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Getting scores from sklearn for training and validation)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


"""
You can switch between training AdaBoostClassifier, RandomForestClassifier, and BaggingClassifier
"""
# Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Enum usage)
class Classifier(Enum):
    ada_boost = 1
    random_forest = 2
    bagging = 3

"""
Argparse for which model you would like to train
"""
# Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Setting up and crafting parse_args)
def parse_args():
    parser = argparse.ArgumentParser(description="Train a malignant/benign tumor diagnoses classifier using analysis on radiology notes.")
    parser.add_argument(
        "--model",
        choices=[c.name for c in Classifier],
        default=Classifier.ada_boost,
        type=str,
        help="Classifier to use",
    )
    return parser.parse_args()


"""
Gets the precision, recall, accuracy, and f1 score based on the prediction
"""
def classifier_scores(truth, prediction):

    accuracy_val = accuracy_score(truth, prediction)
    precision_val = precision_score(truth, prediction)
    recall_val = recall_score(truth, prediction)
    f1_val = f1_score(truth, prediction)

    return (precision_val, recall_val, accuracy_val, f1_val)

# Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Help with plotting confusion matrix)
# Referenced/Used: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
# Referenced/Used: https://pythonguides.com/scikit-learn-confusion-matrix/
# Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
# Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay

"""
Graphs a confusion matrix that counts all the
true positives, true negatives, false positives, and false negatives
results.
"""
def graph_confusion_matrix(truth, prediction, classifier_name="test"):
    results = confusion_matrix(truth, prediction)
    display = ConfusionMatrixDisplay(results, display_labels=["benign", "malignant"])
    display.plot()
    plt.title(f"{classifier_name} classification performance")
    plt.savefig(f"./{classifier_name}.png")


# End of: Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
# End of: Referenced/Used: https://pythonguides.com/scikit-learn-confusion-matrix/
# End of: Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
# End of: Referenced/Used: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
# End of: Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Help with plotting confusion matrix)

# Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
# Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
# Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Help with training and predicting script)


"""
The main script to starting training
"""
def tutorial(classifier_id):
    
    """
    Read all the training features and results into dataframes
    """
    train_features_path = Path("./data/train/features.csv")
    train_results_path = Path("./data/train/results.csv")
# Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Read csv to dataframe)
    train_features_df = pd.read_csv(train_features_path)
    train_results_df = pd.read_csv(train_results_path)
# End of: Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Read csv to dataframe)

# Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Understaning and usage of TfidfVectorizer)
    
    """
    Apply vectorization on the training data.
    Using TfidfVectorizer fit and transform to the training data
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    transformed_train_features_df = vectorizer.fit_transform(train_features_df["notes"])
# End of: Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Understaning and usage of TfidfVectorizer)
# End of: Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    
    """
    Select the classifier to train
    """
    if Classifier.ada_boost.name == classifier_id: 
        classifier = AdaBoostClassifier(random_state=42)
    elif Classifier.random_forest.name == classifier_id:
        classifier = RandomForestClassifier(random_state=42)
    else:
        classifier = BaggingClassifier(random_state=42)
# End of: Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Setting up and crafting parse_args)
# End of: Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Enum usage)

    """
    Fit the training data to the classifer and then call predict on the training data
    for the training predictions
    """
    classifier = classifier.fit(transformed_train_features_df, train_results_df["is_malignant_diagnoses"])

    train_prediction = classifier.predict(transformed_train_features_df)

    """
    Read all the validation features and results into dataframes
    """
    validation_features_path = Path("./data/validation/features.csv")
    validation_results_path = Path("./data/validation/results.csv")

# Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Read csv to dataframe)
    validation_features_df = pd.read_csv(validation_features_path)
    validation_results_df = pd.read_csv(validation_results_path)
# End of: Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Read csv to dataframe)

    """
    Apply vectorization on the validation data.
    """
    transformed_validation_features_df = vectorizer.transform(validation_features_df["notes"])

    """
    Predict with the validation data
    """
    validation_prediction = classifier.predict(transformed_validation_features_df)

    """
    Get scores for training and validation results
    """
    train_scores = classifier_scores(train_results_df["is_malignant_diagnoses"], train_prediction)
    validation_scores = classifier_scores(validation_results_df["is_malignant_diagnoses"], validation_prediction)

    classifier_name = classifier.__class__.__name__
# Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (truncating decimals)


    """
    Print scores
    """
    train_output = f"""Training Scores - {classifier_name}:

Precision: {train_scores[0]:.5f}, Recall: {train_scores[1]:.5f}, Accuracy: {train_scores[2]:.5f}, F1: {train_scores[3]:.5f}
    """
    
    validation_output = f"""Validation Scores - {classifier_name}:

Precision: {validation_scores[0]:.5f}, Recall: {validation_scores[1]:.5f}, Accuracy: {validation_scores[2]:.5f}, F1: {validation_scores[3]:.5f}
    """
# End of: Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (truncating decimals)
# End of: Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Getting scores from sklearn for training and validation)

    print("===================================================================")
    print(train_output)
    print("===================================================================")
    print(validation_output)
    print("===================================================================")

    """
    Get the confusion matrix of the classifier performance for the validation data.
    """
    graph_confusion_matrix(validation_results_df["is_malignant_diagnoses"], validation_prediction, classifier_name)



# End of: Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (Help with training and predicting script)
# End of: Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# End of: Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
# End of: Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html

if __name__ == "__main__":

    args = parse_args()

    tutorial(args.model)

# End of: Referenced/Used: https://codezup.com/natural-language-processing-sentiment-analysis-scikit-learn/
# End of: Referenced/Used: https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
# End of: Referenced/Used: https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/
# End of: Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
# End of: Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
# End of: Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
# End of: Referenced/Used: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# End of: Referenced/Used: https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/
# End of: Referenced/Used AI generated code and info from GPT-4.1 through Github Copilot (General help, Help with parse_args)
# End of: Referenced/Used material from AI healthcare class
# End of: Referenced/Used previous work completed by me for AI healthcare class
