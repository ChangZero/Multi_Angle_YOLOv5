import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'''
__init__(self, ground_truth_dir, predict_dir): with the generator method, initializing the object. Accepts directories of ground truth and predict datasets as arguments.

read_labels(self, file_path): Extract bounding box information of the object by reading the label file of the given path.

bounding_box_to_gaussian(bb): Calculates the mean vector and covariance matrix of Gaussian distribution based on bounding box information.

convert_labels_to_gaussians(self, labels): Receive label information, convert it to Gaussian distribution for each label.

gaussian_wasserstein_distance(mu1, Sigma1, mu2, Sigma2): Calculate the wasserstein distance between two Gaussian distributions.

normalized_gaussian_wasserstein_distance (self, mu1, Sigma1, mu2, sigma2): Returns the calculated wasserstein distance by normalizing it.

estimate_C_from_dataset(self): Estimate the C value based on the average object size of the dataset.

get_nwd_for_file_pair(self, gt_file_path, pred_file_path): Load grount truth and predict datasets using read_labels and Return dataframe

get_nwd_results(self): Calculate the normalized wasserstein distance for ground truth and predict label and return the results to the list.

get_confusion_matrices(self, threshold=0.5): Returns confusion matrix Results by Threshold

plot_confusion_matrices(self, confusion_matrices, sort_descending=True): Create a confusion matrix by class

execute_analysis(self): Execute NGWD process
'''
class GaussianWassersteinDistanceCalculator:
    def __init__(self, ground_truth_dir, predict_dir, n_class=0):
        self.ground_truth_dir = ground_truth_dir
        self.predict_dir = predict_dir
        self.n_class = n_class
        self.C = self.estimate_C_from_dataset()

    def read_labels(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_label, x_center, y_center, width, height = map(float, parts)
                boxes.append((class_label,x_center, y_center, width, height))
        return boxes

    @staticmethod
    def bounding_box_to_gaussian(bb):
        cx, cy, w, h = bb  # center x, center y, width, height
        mu = np.array([cx, cy])
        sigma_x2 = (w / 2) ** 2 / 4
        sigma_y2 = (h / 2) ** 2 / 4
        covariance_matrix = np.array([[sigma_x2, 0], [0, sigma_y2]])
        return mu, covariance_matrix

    def convert_labels_to_gaussians(self, labels):
        gaussians = []
        for label in labels:
            _, x_center, y_center, width, height = label
            mu, sigma = self.bounding_box_to_gaussian((x_center, y_center, width, height))
            gaussians.append((mu, sigma))
        return gaussians

    @staticmethod
    def gaussian_wasserstein_distance(mu1, sigma1, mu2, sigma2):
        mean_diff = np.linalg.norm(mu1 - mu2)
        sigma_half_diff = np.linalg.norm(np.sqrt(sigma1) - np.sqrt(sigma2), 'fro')
        wasserstein_distance = mean_diff**2 + sigma_half_diff**2
        return wasserstein_distance

    def normalized_gaussian_wasserstein_distance(self, mu1, sigma1, mu2, sigma2):
        wasserstein_distance = self.gaussian_wasserstein_distance(mu1, sigma1, mu2, sigma2)
        nwd = np.exp(-np.sqrt(wasserstein_distance) / self.C)
        return nwd

    # Calculate parameter C; C = avg_width * avg_height
    def estimate_C_from_dataset(self):
        label_files = [os.path.join(self.ground_truth_dir, f) for f in os.listdir(self.ground_truth_dir) if f.endswith('.txt')]
        sizes = []
        for file_path in label_files:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, _, _, width, height = parts
                        sizes.append((float(width), float(height)))
        if sizes:
            avg_width, avg_height = np.mean(sizes, axis=0)
            return avg_width * avg_height
        else:
            raise ValueError("No object sizes found in label files.")


    def get_nwd_for_file_pair(self, gt_file_path, pred_file_path):
        gt_labels = self.read_labels(gt_file_path)
        pred_labels = self.read_labels(pred_file_path)

        gt_gaussians = self.convert_labels_to_gaussians(gt_labels)
        pred_gaussians = self.convert_labels_to_gaussians(pred_labels)

        file_results = []
        for gt_label, gt_gaussian in zip(gt_labels, gt_gaussians):
            for pred_label, pred_gaussian in zip(pred_labels, pred_gaussians):
                if gt_label[0] == pred_label[0]:
                    nwd = self.normalized_gaussian_wasserstein_distance(gt_gaussian[0], gt_gaussian[1], pred_gaussian[0], pred_gaussian[1])
                    file_results.append({
                        "file_pair": os.path.basename(gt_file_path) + "-" + os.path.basename(pred_file_path),
                        "class_label": gt_label[0],
                        "x_center": gt_label[1],
                        "y_center": gt_label[2],
                        "width": gt_label[3],
                        "height": gt_label[4],
                        "nwd": nwd
                    })
        file_results_df = pd.DataFrame(file_results)
        return file_results_df



    def get_nwd_results(self):
        all_results_df = pd.DataFrame(columns=["file_pair", "class_label", "x_center", "y_center", "width", "height", "nwd"])
        
        gt_files = sorted(os.listdir(self.ground_truth_dir))
        pred_files = sorted(os.listdir(self.predict_dir))
        
        for gt_file, pred_file in zip(gt_files, pred_files):
            if gt_file.endswith('.txt') and pred_file.endswith('.txt'):
                gt_path = os.path.join(self.ground_truth_dir, gt_file)
                pred_path = os.path.join(self.predict_dir, pred_file)
                file_pair = f"{gt_file}-{pred_file}"
                
                # NWD calculation for file pairs
                file_nwd_df = self.get_nwd_for_file_pair(gt_path, pred_path)
                
                # Add file pair identifier
                file_nwd_df["file_pair"] = file_pair
                
                # Add results to dataframe
                all_results_df = pd.concat([all_results_df, file_nwd_df], ignore_index=True)
                
        return all_results_df
    
    def get_confusion_matrices(self, threshold=0.5):
        self.nwd_results_list['nwd_binary'] = (self.nwd_results_list['nwd'] >= threshold).astype(int)
        confusion_matrices = {}

        # Generate confusion matrix for each class
        for class_label in self.nwd_results_list['class_label'].unique():
            class_rows = self.nwd_results_list[self.nwd_results_list['class_label'] == class_label]
            TP = class_rows['nwd_binary'].sum()
            FP = len(class_rows) - TP
            TN = (self.nwd_results_list['class_label'] != class_label).sum() - FP
            FN = len(class_rows) - TP

            confusion_matrix = pd.DataFrame({
                'Predicted Positive': [TP, FP],
                'Predicted Negative': [FN, TN]
            }, index=['Actual Positive', 'Actual Negative'])

            confusion_matrix_percentage = confusion_matrix / len(self.nwd_results_list) * 100
            confusion_matrices[class_label] = confusion_matrix_percentage

        return confusion_matrices

    def plot_confusion_matrices(self, confusion_matrices, sort_descending=True):
        sorted_confusion_matrices = sorted(confusion_matrices.items(), key=lambda x: x[0], reverse=sort_descending)

        for class_label, conf_matrix in sorted_confusion_matrices:
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues")
            plt.title(f'Confusion Matrix for Class {str(round(class_label))}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()

    def execute_analysis(self):
        self.nwd_results_list = self.get_nwd_results()  # This calculates the NWD results
        confusion_matrices = self.get_confusion_matrices()  # This generates confusion matrices
        self.plot_confusion_matrices(confusion_matrices)  # This plots the confusion matrices

'''
# Usage
ground_truth_directory = ""
predict_directory = ""

calculator = GaussianWassersteinDistanceCalculator(ground_truth_directory, predict_directory, n_class=4)
calculator.execute_analysis() 
'''