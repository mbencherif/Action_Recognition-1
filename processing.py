import cv2
import numpy as np
import os
import tqdm
import argparse
from typing import Tuple, List
from utils import timeit

def get_optical_flow(video : np.array, frame_shape = (224, 224) : Tuple) -> np.array:
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGBA2BGRAY)
        gray_video.append(np.reshape(img, (*frame_shape, 1)))

    flows = []
    for i in range(0, len(video) - 1):
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])

        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)

        flows.append(flow)
    flows.append(np.zeros(shape = (*frame_shape, 2)))

    return np.array(flows, dtype = np.float32)

def video_to_npy(file_path: str, resize = (224, 224) : Tuple) -> np.array:

    assert os.path.isfile(file_path), "ERROR: File not found"
    cap = cv2.VideoCapture(file_path)

    len_frames = int(cap.get(7))

    try:
        frames = []
        for i in range(len_frames - 1):
            _, frame = cap.read()
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (*resize, 3))
            frames.append(frame)
    except:
        print("ERROR: ", file_path, len_frames, i)
    finally:
        frames = np.array(frames)
        cap.release()

    flows = get_optical_flow(frames, frame_shape = resize)
    result = np.zeros((len(flows), *resize, 5))
    result[..., :3] = frames
    result[..., 3:] = flows

    return result


@timeit
def save_to_npy(file_dir: str, save_dir: str, resize : Tuple) -> None:
    assert os.path.exists(file_dir), f"ERROR: {file_dir} not found!"

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    videos = os.listdir(file_dir)
    count = 0
    with open("list_map.txt", "w") as f:
        for v in tqdm.tqdm(videos):
            f.write(f"{v}\t{count}\n")
            data  = video_to_npy(file_path = os.path.join(file_dir, v), resize = resize)
            data  = np.uint8(data)
            count += 1
            np.save(os.path.join(save_dir, str(count) + '.npy'), data)


if __name__ == '__main__':

    parse = argparse.ArgumentParser(description= "Convert video to npy file")
    parse.add_argument("--file_dir", dest= "file_dir", type=str)
    parse.add_argument("--save_dir", dest= "save_dir", type=str)
    parse.add_argument("--frame_shape", dest= "frame_shape", default=(224, 224))


    args = parse.parse_args()
    print(f"Preprocessing: file_dir: {args.file_dir} and save_dir: {args.save_dir}")
    save_to_npy(file_dir = args.file_dir, save_dir= args.save_dir, resize = args.frame_shape)
    print("Done!")
