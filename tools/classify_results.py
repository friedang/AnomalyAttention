import argparse
import json
from typing import Dict, List
from typing import Callable
import tqdm
import os

import numpy as np
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.data_classes import DetectionMetricData


def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
            #    dist_th: float, # TODO [0.5, 1.0, 2.0, 4.0]
               verbose: bool = False) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    # if verbose:
    #     print("Found {} GT of class {} out of {} total across {} samples.".
    #           format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    tokens = [box.sample_token for box in pred_boxes_list]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    # if verbose:
    #     print("Found {} PRED of class {} out of {} total across {} samples.".
    #           format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    # sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    dist_thresholds = [0.5, 1.0, 2.0, 4.0]
    dist_tps = [[], [], [], []]
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    print('Match and accumulate match data.')
    for pred_box in tqdm.tqdm(pred_boxes_list):
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        for pos, th in enumerate(dist_thresholds):

            is_match = min_dist < th
            if is_match:
                dist_tps[pos].append(1)
            else:
                dist_tps[pos].append(0)

        # for distance 4.0
        for j in range(len(dist_tps[0])):
            if not all([dist_tps[i][j] <= dist_tps[i+1][j] for i in range(3)]):
                print(dist_tps[0][j], dist_tps[1][j], dist_tps[2][j], dist_tps[3][j])
        
        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------
    # Accumulate.
    binary_tp_count = tp
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return (DetectionMetricData(binary_tp_count=binary_tp_count,  # TODO add custom class to this script
                               tp=tp[-1],
                               fp=fp[-1],
                               fn=float(npos)-tp[-1],
                               recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'],
                               ),
            tokens, dist_tps)


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def classify_detections(detection_results, nusc, eval_split, result_path, output_dir):
    # Initialize the DetectionEval object
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join('/workspace/CenterPoint/det3d/datasets/nuscenes/', 'AD_detection.json')
    assert os.path.exists(cfg_path), \
        'Requested unknown configuration AD_detection.json'
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    cfg = DetectionConfig.deserialize(data)
    nusc_eval = DetectionEval(nusc, config=cfg, result_path=result_path, eval_set=eval_split, output_dir=output_dir, verbose=False)

    # Run evaluation
    nusc_eval.evaluate()

    # Convert detection_results to EvalBoxes
    pred_boxes = EvalBoxes()
    print('Convert detection_results to EvalBoxes')
    for sample_token, preds in tqdm.tqdm(detection_results['results'].items()):
        for result in preds:
            if sample_token not in pred_boxes.boxes:
                pred_boxes.boxes[sample_token] = []
            pred_boxes.boxes[sample_token].append(DetectionBox(
                sample_token=sample_token,
                translation=result['translation'],
                size=result['size'],
                rotation=result['rotation'],
                velocity=result['velocity'],
                detection_name=result['detection_name'],
                detection_score=result['detection_score'],
                attribute_name=result['attribute_name'],
            ))

    # Get ground truth boxes
    gt_boxes = nusc_eval.gt_boxes

    # Initialize TP classification dict
    tp_classification = {k: {} for k in detection_results['results'].keys()}

    # Iterate through classes
    print('Classifying detection - Iterate through classes')
    for class_name in tqdm.tqdm(nusc_eval.cfg.class_names):
        # Get distance function and threshold for this class
        dist_fcn = nusc_eval.cfg.dist_fcn_callable

        # Accumulate metrics
        _, tokens, dist_tps = accumulate(gt_boxes, pred_boxes, class_name, dist_fcn)

        
        # Update TP classification
        for i, sample_token in enumerate(tokens):
            dist_TP = [dist_tps[j][i] for j in range(4)]
            if class_name not in tp_classification[sample_token]:
                tp_classification[sample_token][class_name] = []
            tp_classification[sample_token][class_name].append(dist_TP)

    
    # Add TP classification to detection results
    print('Add TP classification to detection results')
    for sample_token, preds in tqdm.tqdm(detection_results['results'].items()):
        for result in preds:
            class_name = result['detection_name']
            if class_name in tp_classification[sample_token]:
                result['dist_TP'] = tp_classification[sample_token][class_name].pop(0)
                result['TP'] = 1 if 1 in result['dist_TP'] else 0

                # assert if dist_TP is correct
                if not all([result['dist_TP'][i] <= result['dist_TP'][i+1] for i in range(3)]):
                    print(f"Incorrect distance thresh for {result['dist_TP']}")
                
                # c=1
            else:
                result['dist_TP'] = [0, 0, 0, 0]
                result['TP'] = 0

    tps = [t for d in detection_results['results'].values() for t in d if t['TP']==1]
    fps = [t for d in detection_results['results'].values() for t in d if t['TP']==0]

    print(f"Number of TPs are {len(tps)}")
    print(f"Number of FPs are {len(fps)}")

    return detection_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=None)
    args = parser.parse_args()

    input_file = args.input_file
    output_file = input_file.replace('.json', '_tp.json')
    output_dir = input_file.replace('results.json', '')
    detection_results = load_json(input_file)

    print("TODO need to add this to extraction")

    keys = list(detection_results['results'].keys())
    # key_set = {}
    for k in keys:
        if not detection_results['results'][k]:
            del detection_results['results'][k]
            continue

    # Initialize nuScenes
    from nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot="data/nuScenes", verbose=True)
    
    # Classify detections
    updated_results = classify_detections(detection_results, nusc, 'train', input_file, output_dir)
    # Save the updated JSON file
    save_json(updated_results, output_file)

    print(f"Number of non dummy items in dets is {len([v for values in updated_results['results'].values() for v in values if v['TP'] != -500])}")
    print(f"Number of TP in dets is {len([v for values in updated_results['results'].values() for v in values if v['TP'] == 1])}")
    print(f"Number of FP in dets is {len([v for values in updated_results['results'].values() for v in values if v['TP'] == 0])}")

    print(f"Updated results saved to {output_file}")

if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception, set_trace
    with launch_ipdb_on_exception():
        main()