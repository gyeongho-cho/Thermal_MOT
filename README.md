
# 🔥 Thermal MOT: Multi-Object Tracking on Thermal Images

<div align="center">
<a href="https://eval.ai/web/challenges/challenge-page/2439/overview"><img src="https://img.shields.io/badge/EvalAI-TMOT%20Challenge-blueviolet" alt="EvalAI"/></a>
<a href="https://pbvs-workshop.github.io/"><img src="https://img.shields.io/badge/2025CVPRW-PBVS-navy" alt="Workshop"/></a>
<a href="https://github.com/wassimea/thermalMOT/"><img src="https://img.shields.io/badge/Github-181717?logo=github" alt="Git"/></a>


  
</div>

>This repository contains my approach to the Thermal MOT Challenge 2025, organized under the 21st IEEE Workshop on Perception Beyond the Visible Spectrum (PBVS). It involves training a thermal image object detector and evaluating multiple MOT frameworks. The excellent submissions will be announced at CVPR 2025.

<br><br>

<div align="center">
  <img src="https://pbvs-workshop.github.io/images/logo_25.jpg" alt="Demo" width="70%">
  
  <br><br>
  
  <img src="figs/test_bytetrack_output.gif" alt="Demo" width="50%">
  
</div>



## ⚙️ Getting Start

### 1. Install dependencies
```bash
$ pip install -r requirements.txt
```

---

### 2. Dataset Structure (TMOT Format)

Please download the dataset from the official competition website. [[link](https://eval.ai/web/challenges/challenge-page/2439/overview)]


📂 Please place the dataset in the following directory structure:
```
tmot_dataset_challenge/
├── images/
│   ├── train/
│   │   └── seq000/thermal/*.jpg
│   ├── val/
│   │   └── seq001/thermal/*.jpg
│   └── test/
│       └── seq002/thermal/*.jpg
├── annotations/
│   ├── train/seq000/thermal/COCO/annotations.json
│   ├── val/seq001/thermal/COCO/annotations.json
│   └── ...
```

---

### 3. Training Object Detection Model (YoloV8s)

Create dataset for training yolo. The dataset will be made at 'data/' folder.
```bash
$ python create_dataset_for_yolo.py
```

Train YoloV8 model with Thermal MOT dataset. The weights file will be created in the 'runs' folder.
```bash
$ python yolov8_train.py
```
---

### 4. Running the Tracker
> 📌 **Important:** This section is currently hard-coded and not fully optimized. It is expected to work correctly, but improvements are planned for future updates.


```bash
python test.py --track_thresh 0.3 --match_thresh 0.95 --seq 0
```
- `--match_thresh` : Matching threshold for tracking.
- `--track_thresh`: Detection confidence threshold.
- `--seq`: Sequence index (based on folder order in dataset 0~5).

---

## 📊 Evaluation Metrics
Tracking performance is evaluated using [MOTMetrics](https://github.com/cheind/py-motmetrics):

- MOTA (Multi Object Tracking Accuracy)
- IDF1 (ID F1 Score)
- MOTP (Multi Object Tracking Precision)
- ID Switches
- IDP / IDR

<div align="center">
  
| | MOTA | MOTP | IDF1 | IDP | IDR | RCLL | PRCN
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Bytetrack | **65.90** | **0.2113** | **65.31** | 73.74 | **58.61** | **73.24** | 92.14 |
| OCSORT | 62.94 | 0.1747 | 61.40 | **75.16** | 51.90 | 66.52 | **96.34** |

</div>

---
## 📚 References
- [Enhancing Thermal MOT: A Novel Box Association Method Leveraging Thermal Identity and Motion Similarity](https://arxiv.org/abs/2411.12943)
- [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics)
- [SORT: Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)
- [Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking](https://arxiv.org/abs/2203.14360)
- [Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification](https://arxiv.org/abs/2302.11813)
- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)


---
