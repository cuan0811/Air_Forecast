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


# Tạo nút gốc
root_node = Node()

data = pd.read_excel("D:\Student\project\data-mining\cuan0811-main\Data_Set\data_set_model_processed2.xlsx")
# data = pd.read_excel("D:\Student\project\data-mining\Data_Set\play_Tenis_1.xlsx")
# data = pd.read_excel("D:\Student\project\data-mining\Data_Set\data100.xlsx")

from sklearn.model_selection import train_test_split
data = data.iloc[0:10000]
X = data.drop(columns='Label')
y = data['Label']
# print(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
data_train = pd.concat([X_train, y_train], axis=1)
print(data_train)
decision_tree_algorithm_C45(root_node, data_train, 'Label', min_samples_split=10)
prune_tree(root_node)
# In cây quyết định
print_tree(root_node)

# Dự đoán nhãn cho dữ liệu kiểm thử
predictions = predict(root_node, X_test)
# print("Predictions:", predictions)

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Dự đoán nhãn cho tập kiểm tra
y_pred = clf.predict(X_test)

# Đánh giá độ chính xác của mô hình với mô hình thư viện sklearn
accuracy = accuracy_score(predictions, y_pred)
print("Accuracy:", accuracy)

# data_train = data.iloc[0:1000]
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split
# # print(data)
# data = pd.read_excel("D:/Student/project/data-mining/Data_Set/data.xlsx")
# # data = data.drop(columns=['station', 'No', 'month', 'year', 'day', 'hour', 'wd', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM'])
# # print(data)
# X = data.drop(columns='Label')
# y = data['Label']
# # print(X, y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# data_train = pd.concat([X_train, y_train], axis=1)
# print(data_train)
# decision_tree_algorithm_C45(root_node, data_train, 'Label', min_samples_split=20)
#
# prune_tree(root_node)
# # In cây quyết định
# print_tree(root_node)
#
# # Dự đoán nhãn cho dữ liệu kiểm thử
# predictions = predict(root_node, X_test)
# # print("Predictions:", predictions)
#
# from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier
#
# clf = DecisionTreeClassifier(random_state=42)
# clf.fit(X_train, y_train)
#
# # Dự đoán nhãn cho tập kiểm tra
# y_pred = clf.predict(X_test)
#
# # Đánh giá độ chính xác của mô hình với mô hình thư viện sklearn
# accuracy = accuracy_score(predictions, y_pred)
# print("Accuracy:", accuracy)
#
# data_test = data.iloc[500:2000]
# y_pred = data['Label'].iloc[500:2000]
# # Dự đoán nhãn cho dữ liệu kiểm thử
# predictions = predict(root_node, data_test)
# print("Predictions:", predictions)
#
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(predictions, y_pred)
# print("Accuracy:", accuracy)
# Tải dữ liệu
# data = pd.read_excel("D:/Student/project/data-mining/Data_Set/play_Tenis_1.xlsx")

# Tách dữ liệu thành các thuộc tính và nhãn

#
# # Báo cáo phân loại
# class_report = classification_report(y_test, y_pred)
# print("Classification Report:")
# print(class_report)
#
# # Ma trận nhầm lẫn
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(conf_matrix)
#
# # Biểu đồ ma trận nhầm lẫn
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()
#
# # Vẽ cây quyết định
# plt.figure(figsize=(20,10))
# tree.plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, rounded=True)
# plt.show()


# # So sánh dự đoán với giá trị nhãn thực tế
# actual_labels = data_test['play']
# accuracy = np.mean(predictions == actual_labels)
# print("Accuracy:", accuracy)

# actual_labels = data_test['Label']
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Đánh giá độ chính xác
# accuracy = accuracy_score(actual_labels, predictions)
# print("Accuracy:", accuracy)
#
# # Báo cáo phân loại
# class_report = classification_report(actual_labels, predictions)
# print("Classification Report:")
# print(class_report)
#
# # Ma trận nhầm lẫn
# conf_matrix = confusion_matrix(actual_labels, predictions)
# print("Confusion Matrix:")
# print(conf_matrix)
#
# # Biểu đồ ma trận nhầm lẫn
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()



# Đường cong ROC và AUC
# Note: Đường cong ROC và AUC chỉ áp dụng cho bài toán phân loại nhị phân
# Nếu bạn đang làm việc với nhiều lớp, bạn cần sử dụng một phương pháp khác như one-vs-rest (OvR) hoặc one-vs-one (OvO).
# if len(set(actual_labels)) == 2:  # Nếu chỉ có hai lớp
#     auc_score = roc_auc_score(actual_labels, predictions)
#     print("ROC AUC Score:", auc_score)
#     # Plot ROC Curve
#     fpr, tpr, _ = roc_curve(actual_labels, predictions)
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     plt.show()
# else:
#     print("ROC AUC Score is not applicable for multi-class classification.")

