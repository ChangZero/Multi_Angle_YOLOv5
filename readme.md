# Detection of Engine Hose Misconnections Using Deep Learning-Based Object Recognition

-   [2023 spirng KIIE(Korean Institute of Industrial Engineers) Poster](https://github.com/Dais-lab/Multi_Angle_engine_clamp_detection/blob/main/2023_spirng_KIIE(Korean%20Institute%20of%20Industrial%20Engineers)%20Poster.pdf)
-   [Presentation_pdf](https://github.com/Dais-lab/Multi_Angle_YOLOv5/blob/main/ppt.pdf)

## 1. Project Abstract

### Abstract

In this study, we developed a technology that utilizes deep learning-based object recognition to detect real-time engine hose misconnections caused by worker errors. We trained a model to detect the connection points and tools (wrenches) within the target engine, using CCTV footage capturing the work process. Based on the inferred task completion, the model predicts the status of each connection point. To address the issue of obstructed footage due to the worker's movements, we applied the Multi-Angle Processing technique, which combines video footage captured from different angles.

We used the YOLOv5 model, a one-stage detection model, for object recognition. Experimental results showed that when setting the threshold at 0.5 based on mean average precision (mAP), the results were above 0.995 for each class. By applying the Multi-Angle Processing technique, the overall misconnection rate decreased from 0.58 to 0.14 compared to a single CCTV footage.

The proposed technology in this study can be extended and applied to various manufacturing sites where worker errors leading to work omissions can be detected using CCTV footage.

### Contributors

#### Members

[`Changyeong Kim`](https://github.com/ChangZero)|[`Hyungun Cho`](https://github.com/Chohyungun)|[`Junhyuk Choi`](https://github.com/sxs770)

#### Adviser
[`Sudong Lee`](https://dais.ulsan.ac.kr/)

#### Contribution
- [`Changyeong Kim`](https://github.com/ChangZero) &nbsp; PM• Model Training• ID-fixing• Switch Wrench Engagement• Multi-Angle detection
- [`Hyungun Cho`](https://github.com/Chohyungun) &nbsp; Data Preprocessing• Model Training  
- [`Junhyuk Choi`](https://github.com/sxs770)&nbsp; K-means based Extracting Hole Center•NGWD 

## 2. Tool

-   Anaconda
-   Python3.8
-   Pytorch
-   Pandas
-   Opencv-python
-   Pandas
-   Numpy
-   matplotlib
-   scipy

## 3. Getting Start

### Git clone
```
git clone https://github.com/ChangZero/Multi_Angle_engine_clamp_detection.git
```

### Config
Fill in the multi_angle_detect-config.yaml
```
cam1:
    detect_path: ""
    weights: ""
    video_path: ""
    h_info_path: ""
    conf-thres: "0.65"
    epsilon: "100"
    iou: "0.5"
    wst: "3"

cam2:
    detect_path: ""
    weights: ""
    video_path: ""
    h_info_path: ""
    conf-thres: "0.65"
    epsilon: "100"
    iou: "0.5"
    wst: "3"
```

### Build dockerfile
```
docker build --tag ma-yolo-image .
```

## 4. Equipment & Software
- [OS] : Ubuntu 20.04
- [GPU] : CUDA 11.4, NVIDIA RTX A6000
- [Framework] : Pytorch
- [IDE] : Visual Studio Code
- [Collaboration Tool] : Notion, Discord

## 5. License

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />
