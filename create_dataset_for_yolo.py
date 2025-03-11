import os
import json
import shutil
from tqdm import tqdm

# 기존 데이터 경로
IMAGE_SRC_PATH = "./tmot_dataset_challenge/images"
ANNOTATION_SRC_PATH = "./tmot_dataset_challenge/annotations"

# YOLO 데이터 경로
YOLO_DATA_PATH = "./data"

# COCO 형식 -> YOLO 형식 변환 함수
def convert_coco_to_yolo(anno_data, img_width, img_height):
    yolo_data = []
    for obj in anno_data:
        category_id = obj["category_id"] - 1  # YOLO는 class id 0부터 시작
        x, y, w, h = obj["bbox"]

        # YOLO 포맷으로 변환
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w = w / img_width
        h = h / img_height

        yolo_data.append(f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    
    return yolo_data

# 데이터 변환 실행
for split in ["train", "val"]:
    split_image_src = os.path.join(IMAGE_SRC_PATH, split)
    split_anno_src = os.path.join(ANNOTATION_SRC_PATH, split)
    
    split_image_dest = os.path.join(YOLO_DATA_PATH, split)
    os.makedirs(split_image_dest, exist_ok=True)

    for seq_id in tqdm(os.listdir(split_image_src), desc=f"Processing {split} data"):
        seq_image_path = os.path.join(split_image_src, seq_id, "thermal")
        seq_anno_path = os.path.join(split_anno_src, seq_id, "thermal", "COCO", "annotations.json")

        if not os.path.exists(seq_anno_path):
            continue  # Annotation 파일 없으면 건너뜀

        # Annotation 파일 로드
        with open(seq_anno_path, "r") as f:
            coco_data = json.load(f)

        images_info = {img["id"]: img for img in coco_data["images"]}
        annotations_info = {}
        for anno in coco_data["annotations"]:
            img_id = anno["image_id"]
            if img_id not in annotations_info:
                annotations_info[img_id] = []
            annotations_info[img_id].append(anno)

        # 이미지 및 YOLO annotation 변환
        for img_id, img_info in images_info.items():
            img_filename = img_info["file_name"]
            src_img_path = os.path.join(seq_image_path, img_filename)
            dest_img_path = os.path.join(split_image_dest, img_filename)

            if not os.path.exists(src_img_path):
                continue  # 이미지 파일이 없으면 건너뜀

            # 이미지 복사
            shutil.copy(src_img_path, dest_img_path)

            # Annotation 변환
            yolo_annotations = convert_coco_to_yolo(annotations_info.get(img_id, []), img_info["width"], img_info["height"])
            yolo_anno_path = os.path.join(split_image_dest, img_filename.replace(".png", ".txt"))

            with open(yolo_anno_path, "w") as f:
                f.write("\n".join(yolo_annotations))

# COCO YAML 파일 생성
coco_yaml_path = os.path.join(YOLO_DATA_PATH, "coco.yaml")
with open(coco_yaml_path, "w") as f:
    yaml_content = f"""\
train: {os.path.abspath(os.path.join(YOLO_DATA_PATH, 'train'))}
val: {os.path.abspath(os.path.join(YOLO_DATA_PATH, 'val'))}

nc: {len(coco_data["categories"])}
names: {json.dumps([cat["name"] for cat in coco_data["categories"]])}
"""
    f.write(yaml_content)

print("✅ YOLO 데이터셋 변환 완료!")