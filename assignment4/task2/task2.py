import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import union
from torch import gt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # Find the intersection boundaries
    intersection = [
        max(prediction_box[0], gt_box[0]),
        max(prediction_box[1], gt_box[1]),
        min(prediction_box[2], gt_box[2]),
        min(prediction_box[3], gt_box[3]),
    ]

    # Find the area of the intersection
    # Negative differences means no intercept and are removed by taking the max with 0.0
    intersection_area = max(intersection[2] - intersection[0], 0.0) * max(intersection[3] - intersection[1], 0.0)

    # Compute union
    # Calculate area of gt_box, prediction_box and union
    gt_area = max(gt_box[2] - gt_box[0], 0.0) * max(gt_box[3] - gt_box[1], 0.0)
    prediction_area = max(prediction_box[2] - prediction_box[0], 0.0) * max(prediction_box[3] - prediction_box[1], 0.0)
    union_area = gt_area + prediction_area - intersection_area
    
    # Calculate intersection over union
    iou = intersection_area / union_area

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """

    if num_tp + num_fp == 0:
        return 1
    else:
        return num_tp / (num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0
    else:
        return num_tp / (num_tp + num_fn)


def get_all_box_matches(pred, gt, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    IoUs = []

    for predicted_box_idx, predicted_box in enumerate(pred):
        for gt_box_idx, gt_box in enumerate(gt):
            iou = calculate_iou(predicted_box, gt_box)
            if iou >= iou_threshold:
                IoUs.append([iou, gt_box_idx, predicted_box_idx])

    # Sort all matches on IoU in descending order
    IoUs.sort(key=lambda x: x[0], reverse=True)

    # Find all matches with the highest IoU threshold
    matches = dict()
    for IoU in IoUs:
        if (IoU[1] not in matches.keys()) and (IoU[2] not in matches.values()) and (len(matches.keys()) < len(gt)):
            matches[IoU[1]] = IoU[2]

    return_prediction_boxes = [pred[i] for i in matches.values()]
    return_gt_boxes = [gt[i] for i in matches.keys()]
    
    if (len(return_prediction_boxes) == 0):
        return_prediction_boxes = np.array([])

    if (len(return_gt_boxes) == 0):
        return_gt_boxes = np.array([])

    return return_prediction_boxes, return_gt_boxes


def calculate_individual_image_result(pred, gt, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    matched_prediction_boxes, matched_gt_boxes = get_all_box_matches(pred, gt, iou_threshold)

    tp = len(matched_prediction_boxes)
    fp = len(pred) - tp
    fn = len(gt) - tp

    return {
        "true_pos": tp, 
        "false_pos": fp,
        "false_neg": fn
    }


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """

    num_tp = 0
    num_fp = 0
    num_fn = 0
    for pred, gt in zip(all_prediction_boxes, all_gt_boxes):
        result_dict = calculate_individual_image_result(pred, gt, iou_threshold)
        num_tp += result_dict["true_pos"]
        num_fp += result_dict["false_pos"]
        num_fn += result_dict["false_neg"]

    return (calculate_precision(num_tp, num_fp, num_fn), calculate_recall(num_tp, num_fp, num_fn))


def get_precision_recall_curve(all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
 

    # YOUR CODE HERE
    precisions = []
    recalls = []


    for threshold in confidence_thresholds:
        
        #find all the prediction boxes over a given threshold
        all_prediction_boxes_over_threshold = []
        
        image_index = 0
        
        for image_confidence_scores in confidence_scores:
     
            single_prediction_boxes_over_threshold = []
            
            confidence_score_index = 0

            for confidence_score in image_confidence_scores:
                
                #check if the confidence score of the prediction box for a given image is above the threshold
                if(confidence_score>=threshold):

                    single_prediction_boxes_over_threshold.append(all_prediction_boxes[image_index][confidence_score_index])

                confidence_score_index = confidence_score_index+1

            all_prediction_boxes_over_threshold.append(single_prediction_boxes_over_threshold)

            image_index = image_index + 1
        
        #calculate the precision and recall for a given threshold
        precision, recall = calculate_precision_recall_all_images(all_prediction_boxes_over_threshold, all_gt_boxes, iou_threshold)

        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("task2/precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    precision_sum = 0

    for recall_level in recall_levels:
        max_precision = 0

        for recall, precision in zip(recalls, precisions):
            if (recall >= recall_level) and (precision > max_precision):
                max_precision = precision

        precision_sum += max_precision

    average_precision = precision_sum / len(recall_levels)

    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()

    mean_average_precision(ground_truth_boxes, predicted_boxes)
