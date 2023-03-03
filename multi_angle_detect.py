import subprocess
import yaml
import multiprocessing
from multiprocessing import Process
import json

def cam1(cam1_path, cam1_weights, cam1_video):
    print("cam1 detection start!")
    # time.sleep(5)
    subprocess.run(["python", cam1_path, "--weights", cam1_weights, "--source", cam1_video, "--conf-thres", "0.65"])
    print("cam1 detection end!")
    
def cam2(cam2_path, cam2_weights, cam2_video):
    print("cam2 detection start!")
    # time.sleep(3)
    subprocess.run(["python", cam2_path, "--weights", cam2_weights, "--source", cam2_video, "--conf-thres", "0.65"])
    print("cam2 detection end!")
     
def main():
    with open('./multi_angle_detect-config.yaml', 'r') as f:
        data = yaml.full_load(f)

    cam1_path = data['cam1']['cam1_path']
    cam1_weights = data['cam1']['cam1_weights']
    cam1_video = data['cam1']['cam1_video']
    cam2_path = data['cam2']['cam2_path']
    cam2_weights = data['cam2']['cam2_weights']
    cam2_video =  data['cam2']['cam2_video']
    
    p1 = Process(target=cam1, args=(cam1_path, cam1_weights, cam1_video))
    p2 = Process(target=cam2, args= (cam2_path, cam2_weights, cam2_video))
    p2.start()
    p1.start()
    p1.join()
    p2.join()
    
    cam1_h_status = dict()
    cam2_h_status = dict()
    result = dict()

    filepath1 = "./cam1_h_status.json"
    filepath2 = "./cam2_h_status.json"
    resultfilepath = "./result.json"
    with open(filepath1, 'r+') as file:
        cam1_h_status = json.load(file)

    with open(filepath2, 'r+') as file:
        cam2_h_status = json.load(file)

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
    
    