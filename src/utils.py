import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from datetime import datetime

from sklearn.metrics import classification_report, confusion_matrix

CLASS_LABELS = [
    'Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine',
    'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz'
]


def get_output_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    master_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(master_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_forestcover_data(filepath):
    """ Load and return forest cover dataset """
    print("Loading forest cover dataset...")
    data = pd.read_csv(filepath)

    # defensive check
    if data.isnull().sum().any():
        print("Warning: missing values detected")
    else:
        print("No missing values found")

    # drop correlated feature identified in EDA
    data = data.drop(columns=['Hillshade_3pm'], errors='ignore')

    # extract features and labels
    X = data.drop(columns=['class'])
    y = data['class'] - 1   # convert 1-7 to 0-6 for keras

    print(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")
    return X, y


def evaluate_model(model, X_test, y_test):
    """ Evaluate and print model performance """
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    # classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"\n{'='*50}")
    print("MODEL EVALUATION RESULTS")
    print(f"\n{'='*50}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred_classes, target_names=CLASS_LABELS))

    return y_pred_classes, acc


def save_confusion_matrix(y_test, y_pred_classes, model_name):
    """ Create and save confusion matrix figure """

    # create output directory path
    output_dir = get_output_dir()

    # generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    # create figure with better styling
    plt.figure(figsize=(10, 8))

    # create heatmap
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=CLASS_LABELS,
                yticklabels=CLASS_LABELS)
    plt.title('Baseline Model - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # save figure
    filename = f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # close the figure to free memory
    plt.close()

    # print confusion matrix to console as well
    print(f"\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=CLASS_LABELS, columns=CLASS_LABELS))
    print(f"\nConfusion matrix saved to: {output_path}")


def plot_training_curves(history, model_name):
    """ Create and save training curves figure"""
    # create output directory path
    output_dir = get_output_dir()

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='validation')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='validation')
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_training_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Training curves saved to: {output_path}")


def save_model(model, acc, test_size, random_state, model_name):
    """ Save the trained model with metadata """
    # create output directory
    output_dir = get_output_dir()

    # generate filename with timestamp and parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = model_name.replace(' ', '_').lower()
    model_filename = os.path.join(
        output_dir,
        f"forestcover_{prefix}_acc{acc:.3f}_ts{test_size}_rs{random_state}_{timestamp}.keras"
    )

    # save model
    model.save(model_filename)

    return model_filename
