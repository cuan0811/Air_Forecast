import numpy as np
import pandas as pd


class Node:
    def __init__(self, attribute=None, threshold=None):
        self.attribute = attribute
        self.threshold = threshold
        self.frame = None
        self.children = []
        self.leaf = False
        self.label = None

    def add_frame(self, frame):
        self.frame = frame

def entropy(data: pd.DataFrame, label):
    all_count = len(data)
    class_counts = data[label].value_counts()
    class_probabilities = class_counts / all_count
    # class_probabilities += 1e-10  # To avoid log(0)
    return -np.sum(class_probabilities * np.log2(class_probabilities))


def gain_ratio(data: pd.DataFrame, label):
    features = data.columns.difference([label])
    ret = {}
    curr_entropy = entropy(data, label)
    for feature in features:
        intrinsic_value = None
        best_threshold = None
        if data[feature].dtype == 'object':
            feature_count = data[feature].value_counts(normalize=True)
            feature_entropy = np.sum([entropy(data[data[feature] == value], label) * feature_count[value] for value in feature_count.index])
            intrinsic_value = -np.sum(feature_count * np.log2(feature_count))
        else:
            sorted_data = data.sort_values(by=feature)
            thresholds = sorted_data[feature].unique()
            feature_entropy = float('inf')

            for i in range(1, len(thresholds)):
                threshold = thresholds[i]
                below_threshold = data[data[feature] <= threshold]
                above_threshold = data[data[feature] > threshold]

                if len(below_threshold) > 0 and len(above_threshold) > 0:
                    weighted_entropy = (len(below_threshold) / len(data) * entropy(below_threshold, label) + len(
                        above_threshold) / len(data) * entropy(above_threshold, label))

                    if weighted_entropy < feature_entropy:
                        feature_entropy = weighted_entropy
                        best_threshold = threshold

            if best_threshold is not None:
                below_threshold = data[data[feature] <= best_threshold]
                above_threshold = data[data[feature] > best_threshold]
                if len(below_threshold) != 0 and len(above_threshold) != 0:
                    intrinsic_value = -((len(below_threshold) / len(data)) * np.log2(len(below_threshold) / len(data)) + (len(above_threshold) / len(data)) * np.log2(len(above_threshold) / len(data)))
                else:
                    intrinsic_value = 0
        gain = curr_entropy - feature_entropy
        if intrinsic_value != 0:
            ret[feature] = gain / intrinsic_value

    return ret


def select_best_attribute(data, label):
    gain_ratios = gain_ratio(data, label)
    max_gain_ratio_attr = max(gain_ratios, key=gain_ratios.get)
    return max_gain_ratio_attr


def decision_tree_algorithm_C45(root, data: pd.DataFrame, label, min_samples_split=2):
    if len(data) == 0:
        root.label = "Failure"
        return root

    elif len(set(data[label])) == 1:
        root.leaf = True
        root.label = data[label].iloc[0]
        return root

    elif len(data.columns) == 1 or len( data) < min_samples_split:
        root.leaf = True
        root.label = data[label].mode()[0]
        return root

    else:
        best_attr = select_best_attribute(data, label)
        root.attribute = best_attr

        if data[best_attr].dtype == 'object':
            for attr_val in data[best_attr].unique():
                subset = data[data[best_attr] == attr_val].drop(columns=[best_attr])
                child_node = Node(attribute=best_attr, threshold=attr_val)
                root.children.append(child_node)
                decision_tree_algorithm_C45(child_node, subset, label, min_samples_split)
        else:
            sorted_data = data.sort_values(by=best_attr)
            thresholds = sorted_data[best_attr].unique()
            best_threshold = None
            feature_entropy = float('inf')

            for i in range(1, len(thresholds)):
                threshold = thresholds[i]
                below_threshold = data[data[best_attr] <= threshold]
                above_threshold = data[data[best_attr] > threshold]

                if len(below_threshold) > 0 and len(above_threshold) > 0:
                    weighted_entropy = (len(below_threshold) / len(data) * entropy(below_threshold, label) + len(above_threshold) / len(data) * entropy(above_threshold, label))

                    if weighted_entropy < feature_entropy:
                        feature_entropy = weighted_entropy
                        best_threshold = threshold

            if best_threshold is not None:
                below_threshold = data[data[best_attr] <= best_threshold]
                above_threshold = data[data[best_attr] > best_threshold]
                root.threshold = best_threshold

                child_node_below = Node(attribute=best_attr, threshold=best_threshold)
                root.children.append(child_node_below)
                decision_tree_algorithm_C45(child_node_below, below_threshold, label, min_samples_split)

                child_node_above = Node(attribute=best_attr, threshold=best_threshold)
                root.children.append(child_node_above)
                decision_tree_algorithm_C45(child_node_above, above_threshold, label, min_samples_split)

        return root


def prune_tree(node):
    if not node.leaf:
        for child in node.children:
            prune_tree(child)

        # Kiểm tra xem có thể cắt tỉa nhánh hiện tại không
        if all(child.leaf for child in node.children):  # Nếu tất cả các con đều là nút lá
            # Lấy danh sách các nhãn của các nút lá con
            child_labels = [child.label for child in node.children]
            # Nếu tất cả các nhãn đều giống nhau, có thể thay thế nhánh bằng một nút lá đơn giản
            if len(set(child_labels)) == 1:
                node.leaf = True
                node.label = child_labels[0]
                node.children = []
                node.attribute = None
                node.threshold = None

def print_tree(node, depth=0):
    if node is None:
        return
    indent = "  " * depth
    if node.leaf:
        print(f"{indent}Leaf Node - Label: {node.label}")
    else:
        print(f"{indent}Node - Attribute: {node.attribute}, Threshold: {node.threshold}")
        for child in node.children:
            print_tree(child, depth + 1)
def predict(node, data_test):
    predictions = []
    for _, instance in data_test.iterrows():
        curr_node = node
        while not curr_node.leaf:
            if curr_node.attribute in instance:
                attr_val = instance[curr_node.attribute]
                if isinstance(attr_val, str):
                    found = False
                    for child in curr_node.children:
                        if child.threshold == attr_val:
                            curr_node = child
                            found = True
                            break
                    if not found:
                        break
                else:
                    if attr_val <= curr_node.threshold:
                        curr_node = curr_node.children[0]
                    else:
                        curr_node = curr_node.children[1]
            else:
                break
        predictions.append(curr_node.label)
    return predictions


root_node = Node()

data = pd.read_excel("D:\Student\project\data-mining\cuan0811-main\Data_Set\data_set_model_processed2.xlsx")

from sklearn.model_selection import train_test_split
data = data.iloc[0:10000]
X = data.drop(columns='Label')
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
data_train = pd.concat([X_train, y_train], axis=1)
print(data_train)
decision_tree_algorithm_C45(root_node, data_train, 'Label', min_samples_split=10)
prune_tree(root_node)

print_tree(root_node)

predictions = predict(root_node, X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(predictions, y_test)
print("Accuracy :", accuracy)