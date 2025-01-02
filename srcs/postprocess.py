'''
Non Max Suppression
IOU, Recall, Precision, Find overlap and Average Precisions
Source Code is adapted from github.com/matterport/MaskRCNN

'''

import numpy as np
import torch
from shapely.geometry import Polygon
from typing import List, Tuple

# def convert_format(boxes_array) ->np.ndarray:
#     """
#     :param boxes_array: an array of shape [# bboxes, 4, 2]
#     :return: a numpy array of shapely.geometry.Polygon objects
#     """
#     print(f"METHOD: convert_format: boxes_array shape: {boxes_array.shape}")
#     print(f"METHOD: convert_format: boxes_array type: {type(boxes_array)}")
#     print(f"METHOD: convert_format: boxes_array: {boxes_array}")
#     polygons = [
#         Polygon([(box[i, 0], box[i, 1]) for i in range(4)] + [(box[0, 0], box[0, 1])])
#         for box in boxes_array
#     ]
#     return np.array(polygons)

def convert_format(boxes_array: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    # boxes_array is a numpy array of shape (N, 4, 2)
    polygons = []
    err_idxs = []
    for idx, box in enumerate(boxes_array):
        try: 
            polygon = Polygon([(point[0], point[1]) for point in box] + [(box[0, 0], box[0, 1])])
            polygons.append(polygon)
        except Exception as e:
            print(f"Error converting bbox at index {idx}: {e}")
            err_idxs.append(idx)
                            
    return np.array(polygons), err_idxs

def compute_overlaps(boxes1: np.ndarray, boxes2: np.ndarray):
    """Computes IoU overlaps between two sets of boxes.
    Returns an overlap matrix, which contains the IoU value for each combination of boxes.
    For better performance, pass the largest set first and the smaller second.
    
    Args: 
        boxes1: a numpy array of shape (N, 4, 2)
        boxes2: a numpy array of shape (M, 4, 2)
    Returns:
        overlaps: a numpy array of shape (N, M)
    """
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    
    boxes1, _ = convert_format(boxes1)
    boxes2, _ = convert_format(boxes2)
    # print(f"METHOD: compute_overlaps: boxes1 type: {type(boxes1)}")
    # print(f"METHOD: compute_overlaps: boxes1 shape: {boxes1.shape}")
    # print(f"METHOD: compute_overlaps: boxes2 type: {type(boxes2)}")
    # print(f"METHOD: compute_overlaps: boxes2 shape: {boxes2.shape}")
    overlaps = np.zeros((len(boxes1), len(boxes2)))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1)
    return overlaps


def compute_iou(box: Polygon, boxes: List[Polygon]):
    """Calculates IoU of the given box with the array of the given boxes.
    Note: the areas are passed in rather than calculated here for efficiency. 
    Calculate once in the caller to avoid duplicate work.
    
    Args:
        box: a polygon (shapely.geometry.Polygon)
        boxes: a numpy array of shape (N,), where each member is a shapely.geometry.Polygon
    Returns:
        a numpy array of shape (N,) containing IoU values
    """
    # print(f"METHOD: compute_iou was called")
    # print(f"METHOD: compute_iou: box type: {type(box)}")
    # print(f"METHOD: compute_iou: box type: {type(boxes)}")
    # print(f"METHOD: compute_iou: boxes shape: {boxes.shape}")
    # print(f"METHOD: compute_iou: boxes member type: {type(boxes[0])}")
    # Calculate intersection areas
    # iou = [box.intersection(b).area / box.union(b).area for b in boxes] #NOTE: Zero division error
    iou_lst = []
    for b in boxes:
        intersection = box.intersection(b).area
        union = box.union(b).area
        iou = intersection / union if union > 0 else 0
        iou_lst.append(iou)
    return np.array(iou_lst, dtype=np.float32)


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.
    pred_boxes: a list of predicted Polygons of size N
    gt_boxes: a list of ground truth Polygons of size N
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids

def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    
    Args:
        boxes: numpy array of shape (N, 4, 2)
        scores: numpy array of shape (N,)    
    
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.

    return an numpy array of the positions of picks
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    polygons, err_idexes = convert_format(boxes)

    if len(err_idexes) > 0:
        # save err_indexes to a file
        np.save("err_indexes.npy", err_idexes)
    top = 64
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1][:64]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(polygons[i], polygons[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)

def filter_invalid_bboxes(corners: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters out bounding boxes (4x2 arrays) with any `inf` or `nan` values.

    Args:
        corners: A numpy array of shape (N, 4, 2), where N is the number of bboxes.
        scores: A numpy array of shape (N,) containing the confidence scores of the bboxes.
    Returns:
        A tuple containing the filtered corners and scores.
    """
    # Check if any element in each bbox contains `inf` or `nan`
    valid_mask = ~np.any(np.isnan(corners) | np.isinf(corners), axis=(1, 2))
    
    # Filter corners to only include valid bboxes
    filtered_corners = corners[valid_mask]
    scores = scores[valid_mask]
    return filtered_corners, scores

def filter_pred(config, pred: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shape of pred is [Batch_Size, 15, H, W]
    
    Args: 
        config: a dictionary containing the configuration parameters
        pred: torch.Tensor of shape [1, 15, 200, 175]
    Returns:
        corners: a numpy array of shape (N, 4, 2)
        scores: a numpy array of shape (N,)
    """
    # print(f"METHOD: filter_pred was called")
    # For overfit testing, shape of pred is [1, 15, 200, 175]
    # print(f"METHOD: filter_pred: pred type: {type(pred)}")
    if len(pred.size()) == 4:
        if pred.size(0) == 1:  # change shape to [15, 200, 175]
            pred.squeeze_(0)
        else:
            raise ValueError("Tensor dimension is not right")

    cls_pred = pred[0, ...]
    # print(f"METHOD: filter_pred: cls_pred.shape: {cls_pred.shape}")
    # np.save("cls_pred.npy", cls_pred.cpu().numpy())
    activation = cls_pred > config['cls_threshold']
    num_boxes = int(activation.sum())
    
    if num_boxes == 0:
        print("No bounding box found")
        return [], []
    print(f"METHOD: filter_pred: num_boxes: {num_boxes}")
    corners = torch.zeros((num_boxes, 8))
    for i in range(7, 15):
        corners[:, i - 7] = torch.masked_select(pred[i, ...], activation)
    corners = corners.view(-1, 4, 2).numpy()
    scores = torch.masked_select(cls_pred, activation).cpu().numpy()
    print(f"METHOD: filter_pred: scores shape: {scores.shape}")
    print(f"METHOD: filter_pred: corners shape : {corners.shape}")
    corners, scores = filter_invalid_bboxes(corners, scores)
    print(f"METHOD: filter_pred: num filtered bboxes: {corners.shape[0]}")
    # print(f"METHOD: filter_pred: scores shape(filtered): {scores.shape}")
    # print(f"METHOD: filter_pred: corners shape(filtered) : {corners.shape}")
    # print(f"METHOD: filter_pred: scores type: {type(scores)}")
    # print(f"METHOD: filter_pred: corners.type: {type(corners)}")
    # print(f"METHOD: filter_pred: corners: {corners}")
    # np.save("corners.npy", corners)
    # NMS
    selected_ids = non_max_suppression(corners, scores, config['nms_iou_threshold'])
    corners = corners[selected_ids]
    scores = scores[selected_ids]
    print(f"METHOD: filter_pred: corners shape after NMS: {corners.shape}")
    print(f"METHOD: filter_pred: scores shape after NMS: {scores.shape}")
    return corners, scores


def compute_ap_range(gt_box, gt_class_id,
                     pred_box, pred_class_id, pred_score,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps = \
            compute_ap(gt_box, gt_class_id,
                       pred_box, pred_class_id, pred_score,
                       iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP

def compute_ap(pred_match: np.ndarray, num_gt: int, num_pred: int):
    """ Compute Average Precision at a set IoU threshold (default 0.5).

        Args:
            pred_match: 1-D array. For each predicted box, it has the index of
                        the matched ground truth box.
            num_gt: Number of ground truth boxes
            num_pred: Number of predicted boxes
    """
    print(f"METHOD: compute_ap was called")
    assert num_gt != 0
    # assert num_pred != 0
    
    # Handle case when there are no predictions
    if num_pred == 0:
        print(f"METHOD: compute_ap: No predictions")
        # If there are no predictions, precision is 0 and recall depends on gt
        mAP = 0.0
        precisions = np.array([0,0])
        recalls = np.array([0, 1])  # Recall jumps from 0 to 1 over the recall range
        precision = 0.0
        recall = 0.0
        return mAP, precisions, recalls, precision, recall
    
    tp = (pred_match > -1).sum()
    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(num_pred) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / num_gt

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
    precision = tp / num_pred
    recall = tp / num_gt
    return mAP, precisions, recalls, precision, recall

def compute_matches(gt_boxes: np.ndarray, #label_list
                    pred_boxes: np.ndarray, # corners
                    pred_scores: np.ndarray, # scores
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.
    
    Args:
        gt_boxes: [N, 4, 2] Coordinates of ground truth boxes
        pred_boxes: [N, 4, 2] Coordinates of predicted boxes
        pred_scores: [N,] Confidence scores of predicted boxes
        iou_threshold: Float. IoU threshold to determine a match.
        score_threshold: Float. Score threshold to determine a match.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # print(f"METHOD: compute_matches was called")
    # print(f"METHOD: compute_matches: gt_boxes type: {type(gt_boxes)}")
    # print(f"METHOD: compute_matches: gt_boxes shape: {gt_boxes.shape}")
    # print(f"METHOD: compute_matches: pred_boxes type: {type(pred_boxes)}")
    # print(f"METHOD: compute_matches: pred_boxes shape: {pred_boxes.shape}")
    # print(f"METHOD: compute_matches: pred_scores type: {type(pred_scores)}")
    # print(f"METHOD: compute_matches: pred_scores shape: {pred_scores.shape}")
    if len(pred_scores) == 0:
        return -1 * np.ones([gt_boxes.shape[0]]), np.array([]), np.array([])

    gt_class_ids = np.ones(len(gt_boxes), dtype=int)
    pred_class_ids = np.ones(len(pred_scores), dtype=int)

    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    overlaps = compute_overlaps(pred_boxes, gt_boxes)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0] #np.where returns a tuple (array, ) for 1D np array
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs: 
            # If ground truth box is already matched, go to next one
            if gt_match[j] > 0:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break #NOTE: sorted_ixs is in descending order, so if iou < iou_threshold, all the following ious will be less than iou_threshold
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps
