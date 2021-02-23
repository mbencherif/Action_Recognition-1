from model import *
import cv2
from PIL import Image
import torch
from config import Config
import argparse

class VideoLoader(object):
    
    def __init__(self, path, n_frame):
        
        if os.path.exists(path):
            self.cap = cv2.VideoCapture(path)
        else:
            raise FileNotFoundError
        
        self.pos = self.cap.get(cv2.CAP_PROP_POS_FRAME)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n_frame = n_frame
    def get_frame(self, sample_rate = 1):
        
        frames = []
        for i in range(self.n_frame * sample_rate):
            ret, frame = self.cap.read()
            if not ret:
                return None
            if i % sample_rate == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.pos = self.cap.get(cv2.CAP_PROP_POS_FRAME)
        return np.array(frames)
    
    def get_random_frame(self, stride = 1):
        
        n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = random.randint(0, max(0, n_frames - 1 - (self.n_frame - 1) * stride))
        frames = []
        count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                return None
            count += 1
            if count == start:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if count == start + self.n_frame:
                return frames


    def __del__(self):
        self.cap.release()
        

def resize(frames, size = (224, 224), scale = 1.0):
    _, h, w, _ = frames.shape
    
    
    crop = int(min(h, w) * scale)
    h1 = h // 2 - crop // 2
    h2 = h1 + crop
    w1 = w // 2 - crop // 2
    w2 = w1 + crop
    
    frames = frames[:, h1 : h2, w1 : w2, :]
    
    clip = [Image.fromarray(frame).resize(size, Image.BILINEAR) for frame in frames]      
    return clip

def transforms(pic):
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    n_channels = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], n_channels)
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        img = img.float().div(255)
    for t, m, s in zip(img, [0, 0, 0], [1, 1, 1]):
        t.sub_(m).div_(s)
    return img

def frames_to_tensor(frames):
    
    clip = resize(frames)
    clip = [transforms(frame) for frame in frames]
    return torch.stack(clip).permute(1, 0, 2, 3)        

def sec_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s



class Demo(object):
    
    
    def __init__(self, model_name, path_checkpoint, config):
        
        self.model_name = model_name
        self.path_checkpoint = path_checkpoint
        self.config = config
        self.MODEL_SUPPORT = ['C3D', 'ConvLSTM', 'DenseNet']
        self.model, _ = self.load_model()
        
    def load_model(self):
        
        assert self.model_name in self.MODEL_SUPPORT, "Model not supported!!!!"
        
        if model_name == 'C3D':
            return C3D(self.config)
        elif model_name == 'ConvLSTM':
            return ConvLSTM(self.config)
        elif model_name == 'DenseNet':
            return DenseNet(self.config)
        elif model_name == 'Densenet_lean':
            return Densenet_lean(self.config)
        else:
            raise Exception("Model not supported")
    

    def predict(self, x):
        
        x = x.to(self.config.device)
        
        if len(x.shape)< 5:
            x = x.unsqueeze(0)
        
        y = self.model(x)
        
        _, result = y.topk(1, 1, True)
        
        labels = ["NonFight" if x else "Fight" for x in result]
        
        return labels
    
    
    def run(self, path):
        
        assert os.path.exists(path), raise FileNotFoundError
        
        if self.config.test_type < 2:
            video = VideoLoader(path)
            
            if self.config.test_type == 0:
                frames = video.get_random_frame()    
                if frames is None:
                    return  
                x = frames_to_tensor(frames)
                y = self.predict(x)
                for label in y:
                    print(f"Video has label: {label}")
                    
                    
            elif self.config.test_type == 1:
                while True:
                    frames = video.get_frame()
                    if frames is None:
                        return
                    x = frames_to_tensor(frames)
                    y = self.predict(x)
                    for label in y:
                        if label == "Fight":
                            time = video.pos / video.fps
                            h, m, s = sec_to_hms(time)
                            print("Fight sence at time: %d:%d:%d" %(h, m, s))
        else:
            def get_frame_names(path, n_frame, stride = 1):
                frame_indices = list(sorted(os.listdir(path)))
                video_duration = len(frame_indices)
                clip_duration = n_frame * stride
                center_index = video_duration // 2 
                begin_index = max(0, center_index - (clip_duration // 2))
                end_index = min(begin_index + clip_duration, video_duration)
                out = frame_indices[begin_index : end_index]
                for index in out:
                    if len(out) >= clip_duration:
                        break
                    out.append(index)
                selected_frames = [out[i] for i in range(0, clip_duration, stride)]
                return selected_frames
            
            image_path = get_frame_names(path, n_frame = self.n_frame)
            frames = []
            for img_path in image_path:
                img = Image.open(os.path.join(path, img_path)).convert('RGB')
                x = frames_to_tensor(frames)
                y = self.predict(x)
                for label in y:
                    print(f"Video has label: {label}")                    
            
            
if __name__ == "__main__":
    
    config = Config()
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--SAMPLE_DURATION", dest='sample_duration', default=16, type=int)
    parser.add_argument("--SAMPLE_SIZE", dest='sample_size', default= (224, 224))
    parser.add_argument("--DEVIDE", dest='device', default='cuda')
    parser.add_argument("--MODEL", dest='model', type=str, default='DenseNet_lean', choices=['C3D', 'ConvLSTM', 'DenseNet', 'DenseNet_lean'])
    parser.add_argument("--CHECKPOINT", dest='checkpoint', type=str)
    parser.add_argument("--TEST_TYPE", dest='test_type', type=int, default=0)
    parser.add_argument("--PATH", dest='path', type=str)
    
    args = parser.parse_args()
    config.sample_size = args.sample_size
    config.sample_duration = args.sample_duration
    config.device = args.device
    config.test_type = args.test_type
    app = Demo(model_name = args.model, checkpoint = args.checkpoint, config = config)
    app.run(args.path)
    
    
    
            
        
        
        