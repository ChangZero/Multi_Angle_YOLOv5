# Development of an Automated Inspection System for Hydraulic Control Unit Hose Assembly Process using Deep Learning-based Object Detection

-   In the progress, KCI Paper

## 1. Project Abstract

### Abstract

This study automates quality inspection in hydraulic control device hose assembly using deep learning-based object detection. Human errors in manufacturing contribute significantly to defects, necessitating an automated solution. The proposed system addresses challenges, including distinguishing pre-assembly and assembly completion states, limited detection for small-sized parts, real-time feedback importance, providing hose connection point status, and overcoming blind spots during object detection.
The system utilizes YOLOv5 for object detection, introduces a novel assembly state transition algorithm, and employs multi-camera object detection with Python-based multiprocessing. Experimental results showcase successful state transitions, high small object detection rates, and improved performance in detecting assembly completion through multi-camera analysis. This system effectively minimizes errors in the hydraulic control device hose assembly process.

### Contributors

#### Members

[`Changyeong Kim`](https://github.com/ChangZero)|[`Hyungun Cho`](https://github.com/Chohyungun)|[`Junhyuk Choi`](https://github.com/sxs770)

#### Adviser
[`Sudong Lee`](https://dais.ulsan.ac.kr/)

#### Contribution
- [`Changyeong Kim`](https://github.com/ChangZero) &nbsp; PM• Model Training• ID-fixing• Switch Wrench Engagement• Multi-Angle detection• Presentation
- [`Hyungun Cho`](https://github.com/Chohyungun) &nbsp; K-means based Extracting Hole Center• NGWD   
- [`Junhyuk Choi`](https://github.com/sxs770)&nbsp; Data Preprocessing• Model Training 

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
    detect_path: "" # cam1_detect.py path
    weights: "" # cam1_weights.pt_path
    video_path: "" # cam_video path
    h_info_path: "" # h_info_paht; ex) ./hole_json_file/cam1_h_info.json
    conf-thres: "0.65" # confidence threshold
    epsilon: "100" # Distance from hole to wrench head threshold
    iou: "0.5" # Interaction over Union threshold
    wst: "3" # wrench head stay time

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
Fill in the ./hole_json_file/cam{number}_h_info_json`s hole location infomation
```
{
    "h1": [0, 0],
    "h2": [0, 0],
    "h3": [0, 0],
    "h4": [0, 0],
    "h5": [0, 0],
    "h6": [0, 0],
    "h7": [0, 0]
}
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

## 5. Competitions
* UOU Creative Comprehensive Design Competition(UOU 창의적 종합 설계 경진대회, 2023)
* Engineering FestivalCreative Comprehensive Design Competition(공학페스티벌 창의적 종합 설계 경진대회, 2023)

## 6. Awards
* Encouragement Prize on Engineering FestivalCreative Comprehensive Design Competition, Ministry of Trade, Industry and Energy(MOTIE), Korea Institute for Advancement of technolohy(KIAT), Research & Information Center for innovation Engineering Education(RICE)
* Grand prize on UOU Creative Comprehensive Design Competition, University of Ulsan Engineering Education Innovation Center
## 7. License

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />
