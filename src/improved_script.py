import argparse
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from utils import load_forestcover_data, evaluate_model, save_confusion_matrix, plot_training_curves, save_model, save_scaler


def run_training_pipeline(X, y, epochs=25, batch_size=32, test_size=0.2, random_state=42):
    """
    Full training pipeline including data splitting, scaling,
    class weight computation, model building and training.
    Returns trained model, test data and training history.
    """

    tf.random.set_seed(random_state)

    # split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train
    )

    # scale only continuous features
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

    X_train = ct.fit_transform(X_train)
    X_val = ct.transform(X_val)
    X_test = ct.transform(X_test)

    save_scaler(ct)

    # class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weight_dict)

    # build the model
    model = Sequential()
    model.add(InputLayer(shape=(X_train.shape[1],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(7, activation='softmax'))   # 7 cover types

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # train
    es = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[es, rlr],
        verbose=1
    )

    return model, X_test, y_test, history


def main():
    parser = argparse.ArgumentParser(
        description='Train Forest Cover Classification Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--filepath', type=str, default='../cover_data.csv',
        help='Path to forest cover dataset CSV file'
    )

    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Training batch size'
    )

    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Fraction of data to use for testing (0.0 to 1.0)'
    )

    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Random state for reproducible results'
    )

    args = parser.parse_args()

    print("FOREST COVER CLASSIFICATION - IMPROVED MODEL")
    print("="*50)
    print(f"CSV path: {args.filepath}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")

    try:
        # load data
        X, y = load_forestcover_data(args.filepath)

        # Train model
        model, X_test, y_test, history = run_training_pipeline(
            X, y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            test_size=args.test_size,
            random_state=args.random_state
        )

        # evaluate model
        y_pred_classes, acc = evaluate_model(
            model, X_test, y_test
        )

        # save confusion matrix figure
        save_confusion_matrix(y_test, y_pred_classes, model_name='Improved Model')

        # plot_training curves
        plot_training_curves(history, model_name='Improved Model')

        # save model
        model_path = save_model(
            model, acc, args.test_size, args.random_state, model_name='Improved Model'
        )

        print(f"\n{'='*50}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Final Accuracy: {acc:.4f}")
        print(f"Model saved to: {model_path}")
        print(f"\n{'='*50}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        return 1
    return 0


if __name__ == '__main__':
    exit(main())
