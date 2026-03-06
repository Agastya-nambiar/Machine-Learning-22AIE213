import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ============================================================
# B1 : Entropy Calculation
# ============================================================

def calc_entropy(labels):

    unique_vals, freq = np.unique(labels, return_counts=True)

    probs = freq / freq.sum()

    ent = -np.sum(probs * np.log2(probs))

    return ent


# ============================================================
# B2 : Gini Impurity Calculation
# ============================================================

def calc_gini(labels):

    unique_vals, freq = np.unique(labels, return_counts=True)

    probs = freq / freq.sum()

    gini_value = 1 - np.sum(probs ** 2)

    return gini_value


# ============================================================
# B3 : Information Gain
# ============================================================

def compute_information_gain(features, labels, column_id):

    base_entropy = calc_entropy(labels)

    distinct_vals = np.unique(features[:, column_id])

    weighted_ent = 0

    for val in distinct_vals:

        subset_labels = labels[features[:, column_id] == val]

        portion = len(subset_labels) / len(labels)

        weighted_ent += portion * calc_entropy(subset_labels)

    gain = base_entropy - weighted_ent

    return gain


def select_best_attribute(features, labels):

    gain_list = []

    for col in range(features.shape[1]):

        ig = compute_information_gain(features, labels, col)

        gain_list.append(ig)

    best_col = np.argmax(gain_list)

    return best_col


# ============================================================
# B4 : Equal Width Discretization
# ============================================================

def equal_width_discretize(column_data, num_bins=10):

    minimum = np.min(column_data)

    maximum = np.max(column_data)

    interval = (maximum - minimum) / num_bins

    discretized = np.floor((column_data - minimum) / interval)

    return discretized.astype(int)


# ============================================================
# B5 : Basic Decision Tree Root Selection
# ============================================================

def create_simple_tree(features, labels, feature_list):

    root_index = select_best_attribute(features, labels)

    tree_dict = {
        "root_index": root_index,
        "root_feature": feature_list[root_index]
    }

    return tree_dict


# ============================================================
# B6 : Decision Tree Plot
# ============================================================

def draw_decision_tree(features, labels, feature_list, label_names):

    clf = DecisionTreeClassifier()

    clf.fit(features, labels)

    plt.figure(figsize=(12, 8))

    plot_tree(
        clf,
        feature_names=feature_list,
        class_names=label_names,
        filled=True
    )

    plt.title("Decision Tree Visualization")

    plt.show()


# ============================================================
# B7 : Decision Boundary Plot
# ============================================================

def plot_decision_regions(features, labels, feature_list):

    clf = DecisionTreeClassifier()

    clf.fit(features, labels)

    x_start, x_end = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_start, y_end = features[:, 1].min() - 1, features[:, 1].max() + 1

    grid_x, grid_y = np.meshgrid(
        np.arange(x_start, x_end, 0.02),
        np.arange(y_start, y_end, 0.02)
    )

    predictions = clf.predict(np.c_[grid_x.ravel(), grid_y.ravel()])

    predictions = predictions.reshape(grid_x.shape)

    plt.contourf(grid_x, grid_y, predictions, alpha=0.3)

    plt.scatter(features[:, 0], features[:, 1], c=labels)

    plt.xlabel(feature_list[0])
    plt.ylabel(feature_list[1])

    plt.title("Decision Boundary")

    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # Load dataset
    dataset = pd.read_csv(r"C:\Users\sahil\Downloads\Archive\final_combined_dataset.csv")

    # Drop ID column
    dataset = dataset.drop(columns=["Ward NO"])

    # Handle missing values
    dataset = dataset.fillna(0)

    target_name = "Consumption in ML"

    # Convert target into categorical levels
    target_labels = pd.qcut(dataset[target_name], q=3, labels=[0, 1, 2])

    # Feature matrix
    feature_matrix = dataset.drop(columns=[target_name]).values

    feature_list = dataset.drop(columns=[target_name]).columns

    label_names = ["Low", "Medium", "High"]

    # -------------------------------------------------
    # B1 : Entropy
    # -------------------------------------------------
    print("Entropy:", calc_entropy(target_labels))


    # -------------------------------------------------
    # B2 : Gini
    # -------------------------------------------------
    print("Gini Index:", calc_gini(target_labels))


    # -------------------------------------------------
    # B3 : Best Root Feature
    # -------------------------------------------------
    best_attr = select_best_attribute(feature_matrix, target_labels)

    print("Best Feature for Root:", feature_list[best_attr])


    # -------------------------------------------------
    # B4 : Binning
    # -------------------------------------------------
    discretized_features = feature_matrix.copy()

    discretized_features[:, 0] = equal_width_discretize(feature_matrix[:, 0], num_bins=5)

    print("Equal Width Binning applied on first feature")


    # -------------------------------------------------
    # B5 : Simple Tree
    # -------------------------------------------------
    simple_tree = create_simple_tree(feature_matrix, target_labels, feature_list)

    print("Simple Tree Root:", simple_tree)


    # -------------------------------------------------
    # B6 : Tree Visualization
    # -------------------------------------------------
    draw_decision_tree(feature_matrix, target_labels, feature_list, label_names)


    # -------------------------------------------------
    # B7 : Decision Boundary
    # -------------------------------------------------
    two_feature_data = feature_matrix[:, :2]

    plot_decision_regions(two_feature_data, target_labels, feature_list)