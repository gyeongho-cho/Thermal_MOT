import os
import cv2
import json
import numpy as np
import torch
import mmcv
import time
import motmetrics as mm
from ultralytics import YOLO
from tracker.sort import Sort  # SORT 모듈 import
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from OC_SORT.trackers.ocsort_tracker.ocsort import OCSort # OCSORT
# MOT Evaluator 생성
acc = mm.MOTAccumulator(auto_id=True)

# ID별 색상 매핑을 위한 딕셔너리
color_map = {}

def get_color(obj_id):
    """각 객체 ID에 대해 고유한 색상을 생성"""
    if obj_id not in color_map:
        np.random.seed(obj_id)  # ID를 기반으로 색상 고정
        color_map[obj_id] = tuple(np.random.randint(0, 255, 3).tolist())  # RGB 값 생성
    return color_map[obj_id]


def compute_iou(box1, box2):
    """IoU 계산 함수"""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box_area = (x2 - x1) * (y2 - y1)
    boxg_area = (x2g - x1g) * (y2g - y1g)

    return inter_area / float(box_area + boxg_area - inter_area)

def evaluate_tracking(predictions, gt_bboxes):
    """
    predictions: List of (frame_id, track_id, x1, y1, x2, y2)
    gt_bboxes: List of (frame_id, gt_id, x1, y1, x2, y2)
    """
    start_time = time.time()

    for frame_id in set([p[0] for p in predictions]):
        pred_data = [(p[1], p[2:]) for p in predictions if p[0] == frame_id]
        gt_data = [(g[1], g[2:]) for g in gt_bboxes if g[0] == frame_id]

        pred_ids = [p[0] for p in pred_data]
        pred_boxes = [p[1] for p in pred_data]

        gt_ids = [g[0] for g in gt_data]
        gt_boxes = [g[1] for g in gt_data]

        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))

        for g_idx, g in enumerate(gt_boxes):
            for p_idx, p in enumerate(pred_boxes):
                iou_matrix[g_idx, p_idx] = compute_iou(g, p)

        acc.update(
            gt_ids,  # GT Object IDs
            pred_ids,  # Predictions IDs
            1 - iou_matrix  # IoU Distance
        )

    total_time = time.time() - start_time
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_objects', 'idf1', 'mota', 'motp', 'num_switches'], name='MOT Metrics')

    print()
    print('=' * 20, "MOT Metrics", '=' * 20)
    print(summary)
    print(f"Runtime Performance: {total_time:.2f} seconds")
    print('=' * 50)

# GT 데이터 불러오기
def load_coco_annotations(annotation_path):
    """ COCO GT 데이터를 (frame_id, gt_id, x1, y1, x2, y2) 형식으로 로드 """
    gt_annotations = defaultdict(list)

    if not os.path.exists(annotation_path):
        return gt_annotations

    with open(annotation_path, "r") as f:
        coco_data = json.load(f)

    for anno in coco_data["annotations"]:
        frame_id = anno["image_id"]
        bbox_id = anno["track_id"]
        x, y, w, h = anno["bbox"]
        x2, y2 = x + w, y + h

        gt_annotations[frame_id].append((frame_id, bbox_id, x, y, x2, y2))

    return gt_annotations

def write_results(filename, results):
    """ 트래킹 결과를 파일에 저장 """
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},{label},-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores, track_classes in results:
            for tlwh, track_id, score, track_class in zip(tlwhs, track_ids, scores, track_classes):
                if track_id < 0:
                    continue
                x1, y1, x2, y2 = tlwh
                w = x2-x1
                h = y2-y1
                label = track_class
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2), label=label)
                f.write(line)
# 트래킹 수행
def process(tracker_info, folder, annotation_folder, outfile, mode="test"):
    tracker_type, tracker = tracker_info
    results = []
    imgs = sorted(filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')), os.listdir(folder)), key=lambda x: int(x.split('.')[0]))

    # GT 데이터 로드 (train, val 모드에서만 적용)
    if mode in ["train", "val"]:
        annotation_path = os.path.join(annotation_folder, "COCO", "annotations.json")
        gt_data = load_coco_annotations(annotation_path)
    else:
        gt_data = []

    prog_bar = mmcv.ProgressBar(len(imgs))

    for frame_id, img in enumerate(imgs,1):
        img_path = os.path.join(folder, img)
        frame = cv2.imread(img_path)

        # YOLO 객체 탐지 수행
        detections = []
        results_yolo = model.predict(img_path, conf=0.5, verbose=False)

        for result in results_yolo:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                detections.append([x1, y1, x2, y2, conf])

        # 트래킹 실행
        if len(detections)>0:
            if tracker_type == "sort":
                tracked_objects = tracker.update(np.array(detections))
            elif tracker_type ==  "deepsort":
                detected_objects = [([x1, y1, x2 - x1, y2 - y1], conf, 1) for x1, y1, x2, y2, conf in detections]
                tracked_objects = tracker.update_tracks(detected_objects, frame=frame)
            elif tracker_type == "ocsort":
                detected_objects = torch.tensor([(x1, y1, x2, y2, conf) for x1, y1, x2, y2, conf in detections])
                tracked_objects = tracker.update(detected_objects, img_info=(512,640),img_size=(512,640) )

        online_x1y1x2y2 = []
        online_ids = []
        online_scores = []
        online_cls = []

        for obj in tracked_objects:
            if tracker_type == "sort":
                x1, y1, x2, y2, track_id = map(int, obj)
            elif tracker_type == "deepsort":
                if not obj.is_confirmed():  # 확인되지 않은 트랙 제외
                    continue
                x1, y1, w, h = map(int, obj.to_tlwh())  # bbox 좌표 변환
                x2, y2 = x1 + w, y1 + h
                track_id = int(obj.track_id)
            elif tracker_type == "ocsort":
                x1, y1, x2, y2, track_id = map(int, obj)

            online_x1y1x2y2.append([x1, y1, x2, y2])
            online_ids.append(track_id)
            online_scores.append(1.0)
            online_cls.append(1)
            # 트랙 ID 색상 지정
            color = get_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # GT 표시 (train, val 모드에서만 활성화)
        if mode in ["train", "val"]:
            
            for (frame_id, obj_id, x1, y1, x2, y2) in gt_data[frame_id]:

                color = get_color(obj_id)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # 파란색 GT 박스
                cv2.putText(frame, f"ID: {obj_id}", (int(x1), int(y2 + 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        results.append((frame_id, online_x1y1x2y2, online_ids, online_scores, online_cls))

        # 결과 출력
        cv2.imshow(f"YOLO + SORT Tracking ({mode.upper()})", frame)
        cv2.waitKey(1)

        prog_bar.update()
        # if frame_id >10:break
    # 평가 수행
    if mode in ['train','val']:
        annot = []
        for _, gt in gt_data.items():
            annot+=gt
        evaluate_tracking(results, annot)

    cv2.destroyAllWindows()
    write_results(outfile, results)

if __name__ == "__main__":
    # 데이터 경로 설정
    base_path = "/Users/horang/Downloads/tmot_dataset_challenge/"
    data_folder = os.path.join(base_path, "images")
    annotation_folder = os.path.join(base_path, "annotations")
    out_folder = "/Users/horang/Documents/ai/MOT/results/"

    # 사용할 모드 선택 ('train', 'val', 'test')
    mode = "test"
    tracker_type = "ocsort"

    if mode == "train":
        sequences = [seq for seq in sorted(os.listdir(os.path.join(data_folder, "train"))) if seq[:3]=='seq']
    elif mode == "val":
        sequences = [seq for seq in sorted(os.listdir(os.path.join(data_folder, "val"))) if seq[:3]=='seq']
    elif mode == "test":
        sequences = [seq for seq in sorted(os.listdir(os.path.join(data_folder, "test"))) if seq[:3]=='seq']
    else:
        raise ValueError("모드는 'train', 'val', 'test' 중 하나여야 합니다.")
    
    # YOLO 모델 로드
    model_weights = "./weights/train3/best.pt"
    model = YOLO(model_weights)

    for sequence in sequences:
        if tracker_type == "sort":
            tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)
        elif tracker_type == "deepsort":
            tracker = DeepSort(max_age=70)
        elif tracker_type == "ocsort":
            tracker = OCSort(det_thresh=0.5)

        process(
            (tracker_type, tracker),
            os.path.join(data_folder, mode, sequence, 'thermal'),
            annotation_folder=os.path.join(annotation_folder, mode, sequence, 'thermal'),
            outfile=os.path.join(out_folder, f"{sequence}_thermal.txt"),
            mode=mode
        )