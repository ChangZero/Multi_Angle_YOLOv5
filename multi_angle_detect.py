import subprocess
import yaml
import multiprocessing
from multiprocessing import Process
import json


# 
class CamInfo:
    def __init__(self, data):
        self.detect_path = data['detect_path']
        self.weights = data['weights']
        self.video_path = data['video_path']
        self.h_info_path = data['h_info_path']
        self.conf_thres = data['conf-thres']
        self.epsilon = data['epsilon']
        self.iou = data['iou']
        self.wst = data['wst']

# Run the process
def exec_thread(cam):
    print("detection start!")
    subprocess.run(["python", cam.detect_path, "--weights", cam.weights, "--source", cam.video_path, "--h_info_path", cam.h_info_path, "--conf-thres", cam.conf_thres, "--epsilon", cam.epsilon, "--iou", cam.iou, "--wst", cam.wst])
    print("detection end!")
    
    
def main():
    with open('./multi_angle_detect-config.yaml', 'r') as f:
        data = yaml.full_load(f)

    cam1_data = data['cam1']
    cam2_data = data['cam2']
    
    cam1 = CamInfo(cam1_data)
    cam2 = CamInfo(cam2_data)

    try:
        p1 = Process(target=exec_thread, args=(cam1, ))
        p2 = Process(target=exec_thread, args=(cam2, ))
        p2.start()
        p1.start()
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("KeyboradInterrupt exception is caught")
    finally:
        cam1_h_status = dict()
        cam2_h_status = dict()
        result = dict()

        filepath1 = "./hole_json_file/cam1_h_status.json"
        filepath2 = "./hole_json/file/cam2_h_status.json"
        resultfilepath = "./result.json"
        with open(filepath1, 'r+') as file:
            cam1_h_status = json.load(file)

        with open(filepath2, 'r+') as file:
            cam2_h_status = json.load(file)

        # Merge the results
        h_idx = 0
        for cam1_h, cam2_h in zip(cam1_h_status.values(), cam2_h_status.values()):
            h_idx += 1
            key = f'h{h_idx}'
            result[key] = max(cam1_h, cam2_h)

        print(result)
        with open(resultfilepath, 'w', encoding='utf-8') as file:
            json.dump(result, file, indent='\t')

    
if __name__=='__main__':
    multiprocessing.set_start_method("spawn")
    main()
    
    