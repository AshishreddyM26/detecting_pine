# -- Install Dependencies

import ultralytics 
from ultralytics import YOLO

from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import math
import os 


# -- Utility class function


class det_utilities:
    
    def __init__(self):
        pass

    def process_tensors(self, tensor):
        """
        Process the tensor results from the YOLO model.

        Parameters:
            tensor: Tensor results from the YOLO model.

        Returns:
            list: Processed list of frame annotations.
        """
        processed_list = []
        for i in range(29):  
            frame_list = []
            frame = i + 1  # Frame number
            
            for j in range(len(tensor[i].boxes.xywh)):
                # frame_id = int(tensor[i].boxes.id[j])  # ID of each bbox in frame 'i'

                x_center, y_center, w, h = map(float, tensor[i].boxes.xywh[j][:])  # x, y, w, h
                conf = float(tensor[i].boxes.conf[j])  # Confidence
                x_min, y_min = x_center - (w / 2), y_center - (h / 2)
                if y_center >= 660:
                    frame_list.append((frame, 0, np.round(x_min, 4), np.round(y_min, 4),                   # -- include frame_id - id of the bbox if necessary, usually helps in tracking
                                       np.round(w, 4), np.round(h, 4), np.round(conf, 3), -1, -1, -1))
            
            processed_list.append(frame_list)
        return processed_list
    
    def annotations_to_text(self, tensor_list, output_path):
        """
        Convert tensor annotations to text and save to the output path.

        Parameters:
            tensor_list (list): List of processed tensor annotations.
            output_path (str): Path to save the text annotations.

        Returns:
            None
        """
        
        with open(output_path, 'w') as file:
            for frame_list in tensor_list:
                for annotation in frame_list:
                    file.write(','.join(map(str, annotation)) + '\n')


    def detect_and_process(self, weight_file, data_source, output_dir, project, file_name):
        """
        Detect objects in the data source using the YOLO model and process the detection results.

        Parameters:
            weight_file (str): Path to the YOLO weight file.
            data_source (str): Path to the data source (e.g., video file, image folder).
            output_path (str): Path to save the processed annotations.
            project (str): Project name or identifier.

        Returns:
            None
        """
        model = YOLO(weight_file)
        det_results = model.predict(
            source=data_source, iou=0.5, line_width=2, save_frames=True, save=True,
            show_labels=True, show_conf=True, conf=0.25, save_txt=True, show=False, project=project
        )

        tensor_list = self.process_tensors(det_results)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, file_name)
        self.annotations_to_text(tensor_list, output_file)
    
    
    def process_bounding_boxes(self, ground_truth_file, prediction_file):
        """
        Load and process bounding boxes from ground truth and prediction files.

        Parameters:
            ground_truth_file (str): Path to the ground truth file.
            prediction_file (str): Path to the prediction file.

        Returns:
            tuple: DataFrames for ground truth and predictions with additional columns.
        """
        gt_labels = pd.read_csv(ground_truth_file, sep=',', header=None, 
                                names=['frame', 'id', 'x1', 'y1', 'width', 'height', 'conf', 'x', 'y', 'z'])
        pred_labels = pd.read_csv(prediction_file, sep=',', header=None, 
                                  names=['frame', 'id', 'x1', 'y1', 'width', 'height', 'conf', 'x', 'y', 'z'])

        for df in [gt_labels, pred_labels]:
            df['x2'] = df['x1'] + df['width']
            df['y2'] = df['y1'] + df['height']
            df['CenterX'] = (df['x1'] + df['x2']) / 2
            df['CenterY'] = (df['y1'] + df['y2']) / 2

        return gt_labels, pred_labels

    def process_and_match_boxes(self, gt_df, pred_df):
        """
        Process and match bounding boxes using Intersection over Union (IoU).

        Parameters:
            gt_df (DataFrame): Ground truth DataFrame.
            pred_df (DataFrame): Prediction DataFrame.

        Returns:
            DataFrame: Ground truth DataFrame with matched IoU scores.
        """
        def calculate_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            iou = interArea / float(boxAArea + boxBArea - interArea)
            return iou

        gt_boxes = gt_df[['x1', 'y1', 'x2', 'y2']].values
        pred_boxes = pred_df[['x1', 'y1', 'x2', 'y2']].values

        original_iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i in range(len(gt_boxes)):
            for j in range(len(pred_boxes)):
                original_iou_matrix[i, j] = calculate_iou(gt_boxes[i], pred_boxes[j])

        max_boxes = max(len(gt_boxes), len(pred_boxes))
        extended_iou_matrix = np.full((max_boxes, max_boxes), -1e-5)
        extended_iou_matrix[:len(gt_boxes), :len(pred_boxes)] = original_iou_matrix

        gt_indices, pred_indices = linear_sum_assignment(-extended_iou_matrix)

        valid_indices = (gt_indices < len(gt_boxes)) & (pred_indices < len(pred_boxes))
        valid_gt_indices = gt_indices[valid_indices]
        valid_pred_indices = pred_indices[valid_indices]

        matched_iou_scores = original_iou_matrix[valid_gt_indices, valid_pred_indices]

        matched_data = pd.DataFrame({
            'IoU': matched_iou_scores
        }, index=valid_gt_indices)

        gt_df = gt_df.merge(matched_data, left_index=True, right_index=True, how='left')
        return gt_df

    def fill_nan_values(self, df_list):
        """
        Fill NaN values in a list of DataFrames.

        Parameters:
            df_list (list): List of DataFrames to process.

        Returns:
            None
        """
        for i, df in enumerate(df_list):
            if df.isna().any().any() == False:
                print(f'No NaN {i}')
            else:
                df.fillna(0.0, inplace=True)
                print(f'NaN values are converted in {i}')

    def calculate_viewing_angles(self, df, x_center, y_center, happ, vapp):
        """
        Calculate viewing angles for bounding boxes.

        Parameters:
            df (DataFrame): DataFrame containing bounding box data.
            x_center (float): X-coordinate of the image center.
            y_center (float): Y-coordinate of the image center.
            happ (float): Horizontal angular pixel size.
            vapp (float): Vertical angular pixel size.

        Returns:
            pd.Series: Series with calculated viewing angles.
        """
        xd = df['CenterX'] - x_center
        yd = y_center - df['CenterY']

        theta_h = xd * happ
        theta_v = yd * vapp

        combined_oblique_radians = np.arctan(np.sqrt(np.tan(theta_h)**2 + np.tan(theta_v)**2))

        return pd.Series({
            'oblique_angle': np.degrees(combined_oblique_radians),
            'horiz_angle': np.degrees(theta_h),
            'vertical_angle': np.degrees(theta_v)
        })

    def apply_angles_to_dataframe(self, df, hfov_degrees, vfov_degrees, width, height):
        """
        Apply viewing angles to a DataFrame based on camera parameters.

        Parameters:
            df (DataFrame): DataFrame to modify.
            hfov_degrees (float): Horizontal field of view in degrees.
            vfov_degrees (float): Vertical field of view in degrees.
            width (int): Width of the image.
            height (int): Height of the image.

        Returns:
            DataFrame: Modified DataFrame with viewing angles.
        """
        happ = math.radians(hfov_degrees) / width
        vapp = math.radians(vfov_degrees) / height
        x_center = width / 2
        y_center = height / 2

        df[['oblique_va', 'hor_va', 'vert_va']] = df.apply(
            self.calculate_viewing_angles, axis=1, args=(x_center, y_center, happ, vapp))
        return df

    def processing_dfs(self, ground_truth_files, prediction_files, output_paths):
        """
        Process ground truth and prediction files and save results.

        Parameters:
            ground_truth_files (list): List of paths to ground truth files.
            prediction_files (list): List of paths to prediction files.
            output_paths (list): List of output paths for saving results.

        Returns:
            None
        """
        for gt_file, pred_file, output_prefix in zip(ground_truth_files, prediction_files, output_paths):
            gt_df, pred_df = self.process_bounding_boxes(gt_file, pred_file)
            matched_gt_df = self.process_and_match_boxes(gt_df, pred_df)
            
            camera1_settings = {'hfov_degrees': 118, 'vfov_degrees': 69.2, 'width': 3840, 'height': 2160}
            camera2_settings = {'hfov_degrees': 80, 'vfov_degrees': 50, 'width': 1920, 'height': 1080}
            
            self.fill_nan_values([matched_gt_df, pred_df])
            matched_gt_df = self.apply_angles_to_dataframe(matched_gt_df, **camera1_settings) # -- for whitecity 
            
            matched_gt_df.to_csv(f'{output_prefix}_gt.csv', index=False)
            pred_df.to_csv(f'{output_prefix}_pred.csv', index=False)
            print('Done...')

    # -- Dividing the dataframe into rows depending upon its data collected (from setA or setB) 
    
    def rows_setA(self, wc_gt):
        """
        Divide the data into rows based on specified conditions for set A.

        Parameters:
            wc_gt (DataFrame): DataFrame containing the data to be divided.

        Returns:
            list: List of DataFrames, each representing a row in set A.
        """
        row1 = wc_gt[(wc_gt['CenterX'] >= 350) & (wc_gt['CenterX'] <= 725)]
        row2 = wc_gt[(wc_gt['CenterX'] >= 705) & (wc_gt['CenterX'] <= 1100)]
        row3 = wc_gt[(wc_gt['CenterX'] >= 1100) & (wc_gt['CenterX'] <= 1600)]
        row4 = wc_gt[(wc_gt['CenterX'] >= 1600) & (wc_gt['CenterX'] <= 2150)]
        row5 = wc_gt[(wc_gt['CenterX'] >= 2150) & (wc_gt['CenterX'] <= 2600)]
        row6 = wc_gt[(wc_gt['CenterX'] >= 2700) & (wc_gt['CenterX'] <= 3050)]
        row7 = wc_gt[
            ((wc_gt['CenterX'] >= 3050) & (wc_gt['CenterX'] <= 3390) & (wc_gt['CenterY'] <= 2160) & (wc_gt['CenterY'] > 1500)) |
            ((wc_gt['CenterX'] >= 3050) & (wc_gt['CenterX'] <= 3500) & (wc_gt['CenterY'] <= 1500) & (wc_gt['CenterY'] > 0))
        ]
        row8 = wc_gt[
            ((wc_gt['CenterX'] >= 3350) & (wc_gt['CenterX'] <= 3840) & (wc_gt['CenterY'] <= 2160) & (wc_gt['CenterY'] > 1500)) |
            ((wc_gt['CenterX'] >= 3490) & (wc_gt['CenterX'] <= 3840) & (wc_gt['CenterY'] <= 1500) & (wc_gt['CenterY'] > 0))
        ]

        return [row1, row2, row3, row4, row5, row6, row7, row8]


    def rows_setB(self, wc_gt):
        """
        Divide the data into rows based on specified conditions for set B.

        Parameters:
            wc_gt (DataFrame): DataFrame containing the data to be divided.

        Returns:
            list: List of DataFrames, each representing a row in set B.
        """
        row1 = wc_gt[(wc_gt['CenterX'] >= 350) & (wc_gt['CenterX'] <= 725)]
        row2 = wc_gt[(wc_gt['CenterX'] >= 705) & (wc_gt['CenterX'] <= 1100)]
        row3 = wc_gt[(wc_gt['CenterX'] >= 1100) & (wc_gt['CenterX'] <= 1600)]
        row4 = wc_gt[(wc_gt['CenterX'] >= 1600) & (wc_gt['CenterX'] <= 2150)]
        row5 = wc_gt[(wc_gt['CenterX'] >= 2150) & (wc_gt['CenterX'] <= 2600)]
        row6 = wc_gt[(wc_gt['CenterX'] >= 2700) & (wc_gt['CenterX'] <= 3050)]
        row7 = wc_gt[
            ((wc_gt['CenterX'] >= 3050) & (wc_gt['CenterX'] <= 3390) & (wc_gt['CenterY'] <= 2160) & (wc_gt['CenterY'] > 1500)) |
            ((wc_gt['CenterX'] >= 3050) & (wc_gt['CenterX'] <= 3500) & (wc_gt['CenterY'] <= 1500) & (wc_gt['CenterY'] > 0))
        ]
        row8 = wc_gt[
            ((wc_gt['CenterX'] >= 3350) & (wc_gt['CenterX'] <= 3840) & (wc_gt['CenterY'] <= 2160) & (wc_gt['CenterY'] > 1500)) |
            ((wc_gt['CenterX'] >= 3490) & (wc_gt['CenterX'] <= 3840) & (wc_gt['CenterY'] <= 1500) & (wc_gt['CenterY'] > 0))
        ]

        return [row1, row2, row3, row4, row5, row6, row7, row8]
    
    
    def plot_mean_iou_heatmap(self, dfs):
        
        """
        Plots a heatmap of the mean IoU values.

        Parameters:
        dfs (list of DataFrames): List of DataFrames containing 'Y' and 'IoU' columns.

        Returns:
        None
        """
        # Initialize the mean IoU matrix
        mean_iou_matrix = np.zeros((3, len(dfs)))
        height = 2160 / 3

        # Calculate mean IoU for each section and each DataFrame
        for col_index, df in enumerate(dfs):
            mean_iou_matrix[0, col_index] = df[df['CenterY'] < height]['IoU'].mean()  # top section
            mean_iou_matrix[1, col_index] = df[(df['CenterY'] > height) & (df['CenterY'] <= height*2)]['IoU'].mean()  # middle section
            mean_iou_matrix[2, col_index] = df[df['CenterY'] > height*2]['IoU'].mean()  # bottom section

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.heatmap(mean_iou_matrix, annot=True, fmt=".2f", cmap='jet',
                    xticklabels=[f'Row {i+1}' for i in range(len(dfs))],
                    yticklabels=['Top', 'Middle', 'Bottom'], vmin=0, vmax=1, cbar_kws={'label': 'IoU'},
                    ax=ax)
        ax.set_title('Mean IoU')
        
        plt.tight_layout()
        plt.show()
        
        
    def plot_iou_scores(self, df, title):
        """
        Plots bounding box centers colored by IoU scores.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing columns 'CenterX', 'CenterY', and 'IoU'.
        title (str): Title of the plot.
        """
        
        plt.figure(figsize=(16, 6))
        plt.scatter(df['CenterX'], df['CenterY'], alpha=0.7, c=df['IoU'], cmap='seismic', vmin=0, vmax=1, s=10)
        plt.colorbar(label='IoU Scores')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim((0, 3840))
        plt.ylim((2160, 0))
        plt.title(title)
        plt.show()


# -- add the significance test codes here -- then we are done for the step 1

    def significance_test(self, dfs):
        """
        Divide dataframes into segments and calculate correlation between the number of bounding boxes and mean IoU for each segment.

        Parameters:
            dfs (list): List of DataFrames to be segmented.

        Returns:
            tuple: DataFrame with mean IoU for each segment, correlation coefficient, and p-value.
        """
        segment_means_list = []
        
        for df in dfs:
            segments = {'Top': df[(df['CenterY'] >= 0) & (df['CenterY'] < 720)],
                        'Middle': df[(df['CenterY'] >= 720) & (df['CenterY'] < 1440)],
                        'Bottom': df[(df['CenterY'] >= 1440) & (df['CenterY'] < 2160)]}

            segment_means = {'Segment': [], 'Number_of_BBoxes': [], 'Mean_IoU': []}
            
            for segment, segment_df in segments.items():
                if not segment_df.empty:
                    segment_means['Segment'].append(segment)
                    segment_means['Number_of_BBoxes'].append(len(segment_df))
                    segment_means['Mean_IoU'].append(segment_df['IoU'].mean())

            segment_means_list.append(pd.DataFrame(segment_means))
        
        combined_results = pd.concat(segment_means_list)
        
        # Perform correlation analysis
        correlation, p_value = stats.pearsonr(combined_results['Number_of_BBoxes'], combined_results['Mean_IoU'])
        return combined_results, correlation, p_value