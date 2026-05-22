import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score

from tensorflow.keras.models import load_model

from utils import get_output_dir, load_forestcover_data, save_confusion_matrix


def preprocess_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split and scale data — identical pipeline to scripts 1 & 2"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train
    )

    continuous_features = [
        'Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon',
        'Horizontal_Distance_To_Fire_Points'
    ]

    ct = ColumnTransformer(
        [('numeric', StandardScaler(), continuous_features)],
        remainder='passthrough'
    )

    ct.fit(X_train)
    X_val = ct.transform(X_val)
    X_test = ct.transform(X_test)

    return X_val, y_val, X_test, y_test


def evaluate_predictions(y_test, predictions, model_name, class_labels):
    """Evaluate and print classification report for a set of predictions"""
    pred_classes = np.argmax(predictions, axis=1)

    print(f"\n{'='*50}")
    print(f"{model_name} RESULTS")
    print(f"{'='*50}")
    print(classification_report(y_test, pred_classes, target_names=class_labels))

    return pred_classes


def plot_ensemble_comparison(results):
    """Plot accuracy comparison across all models"""
    output_dir = get_output_dir()

    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    macro_f1s = [results[name]['macro_f1'] for name in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # accuracy comparison
    axes[0].bar(model_names, accuracies, color='steelblue', edgecolor='black')
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(max(0.0, min(accuracies) - 0.05), 1.0)
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')

    # macro F1 comparison
    axes[1].bar(model_names, macro_f1s, color='coral', edgecolor='black')
    axes[1].set_title('Macro F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Macro F1')
    axes[1].set_ylim(0.6, 1.0)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(macro_f1s):
        axes[1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ensemble_model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Comparison plot saved to: {output_path}")


def optimize_ensemble_weights(pred_1, pred_2, y_val):
    """Find optimal ensemble weights using scipy minimize"""
    def objective(weights):
        # normalize weights to sum to 1
        w = weights / weights.sum()
        ensemble = w[0] * pred_1 + w[1] * pred_2
        pred_classes = np.argmax(ensemble, axis=1)
        return -f1_score(y_val, pred_classes, average='macro')

    from scipy.optimize import minimize
    result = minimize(
        objective,
        x0=[0.5, 0.5],
        method='L-BFGS-B',
        bounds=[(0.0, 1.0), (0.0, 1.0)]
    )
    optimal_weights = result.x / result.x.sum()
    print(f"Optimal weights: Model 1={optimal_weights[0]:.3f}, Model 2={optimal_weights[1]:.3f}")
    return optimal_weights


def main():
    parser = argparse.ArgumentParser(
        description='Forest Cover Ensemble Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--filepath', type=str, default='../cover_data.csv',
                        help='Path to forest cover dataset CSV file')
    parser.add_argument('--baseline-model', type=str, required=True,
                        help='Path to saved baseline model .h5 file')
    parser.add_argument('--improved-model', type=str, required=True,
                        help='Path to saved improved model .h5 file')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Fraction of training data reserved for weight optimisation')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')

    args = parser.parse_args()

    tf.random.set_seed(args.random_state)

    print("FOREST COVER CLASSIFICATION - ENSEMBLE MODEL")
    print("=" * 50)

    class_labels = [
        'Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine',
        'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz'
    ]

    try:
        output_dir = get_output_dir()

        # load and preprocess data
        X, y = load_forestcover_data(args.filepath)
        X_val, y_val, X_test, y_test = preprocess_data(
            X, y,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state
        )

        # load saved models
        print("\nLoading saved models...")
        model_1 = load_model(args.baseline_model)
        model_2 = load_model(args.improved_model)
        print("Models loaded successfully")

        # optimise weights on validation set — no test data involved
        print("\nOptimising ensemble weights on validation set...")
        val_pred_1 = model_1.predict(X_val)
        val_pred_2 = model_2.predict(X_val)
        optimal_weights = optimize_ensemble_weights(val_pred_1, val_pred_2, y_val)

        # get test-set predictions
        print("\nGenerating test predictions...")
        pred_1 = model_1.predict(X_test)
        pred_2 = model_2.predict(X_test)

        # ensemble predictions
        avg_ensemble = (pred_1 + pred_2) / 2
        weighted_ensemble = optimal_weights[0] * pred_1 + optimal_weights[1] * pred_2

        # store results for comparison plot
        results = {}

        # evaluate all four
        for name, preds in [
            ('Baseline Model', pred_1),
            ('Improved Model', pred_2),
            ('Ensemble Average', avg_ensemble),
            ('Ensemble Weighted', weighted_ensemble)
        ]:
            pred_classes = evaluate_predictions(y_test, preds, name, class_labels)

            # extract accuracy and macro f1 for comparison plot
            results[name] = {
                'accuracy': accuracy_score(y_test, pred_classes),
                'macro_f1': f1_score(y_test, pred_classes, average='macro'),
                'pred_classes': pred_classes
            }

        # identify best model
        best_name = max(results, key=lambda x: results[x]['macro_f1'])
        best_preds = results[best_name]['pred_classes']

        print(f"\n{'='*50}")
        print(f"BEST MODEL: {best_name}")
        print(f"Accuracy: {results[best_name]['accuracy']:.4f}")
        print(f"Macro F1: {results[best_name]['macro_f1']:.4f}")

        # save confusion matrix for best model
        save_confusion_matrix(y_test, best_preds, model_name=best_name)

        # Save predictions
        np.save('best_predictions.npy', best_preds)
        pd.DataFrame({
            'true': y_test.values if hasattr(y_test, 'values') else y_test,
            'predicted': best_preds
        }).to_csv(os.path.join(output_dir, 'best_predictions.csv'), index=False)
        print(f"Predictions saved to: {output_dir}")

        # plot comparison across all models
        plot_ensemble_comparison(results)

        # Only ensemble if baseline is competitive
        if results['Baseline Model']['macro_f1'] > 0.95 * results['Improved Model']['macro_f1']:
            print("\nEnsembling may provide marginal benefit...")
        else:
            print("\nImproved model is significantly better. Ensemble not recommended.")

        print(f"\n{'='*50}")
        print("ENSEMBLE EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*50}")

    except Exception as e:
        print(f"Error during ensemble evaluation: {str(e)}")
        return 1
    return 0


if __name__ == '__main__':
    exit(main())
