from ultralytics import YOLO
import cv2
from glob import glob
import os
model = YOLO("yolov8s.pt") # 원하는 크기 모델 입력(n ~ x)

base_path = "./tmot_dataset_challenge/"
# image_list = glob(os.path.join(base_path, 'images','train','*','thermal','*.png'))
seq_list = glob(os.path.join(base_path, 'images','train','*'))
ann_list = glob(os.path.join(base_path, 'annotations','train','*','thermal','COCO','*'))
# Train the model on Apple silicon chip (M1/M2/M3/M4)
results = model.train(data="./data/coco.yaml", epochs=2, imgsz=640, device="mps")


for seq in seq_list:
    image_list = sorted(glob(os.path.join(seq,'thermal','*.png')))
    for img in image_list:
        print(img)
        result = model.predict(img, save=True, conf=0.5)
        plots = result[0].plot()
        cv2.imshow("plot", plots)
        cv2.waitKey(1)
cv2.destroyAllWindows()