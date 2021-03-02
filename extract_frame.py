import os 
import cv2 

try:
    import ffmpeg 
    
except:
    os.system("pip install ffmpeg-python")
    import ffmpeg
    
OUT_DIR =  'output'

def logs(str):
    with open("logs.txt", 'a') as f:
        f.write(str + "\n")

def convert_to_mp4(in_file):
    name, ext = os.path.splitext(in_file)
    out_name = name + '.mp4'
    try:
        ffmpeg.input(in_file).output(out_name).run()
    except:
        logs(in_file)
        return None
    return out_name



def extract_frame(mp4_file, limit_frames= 200):
    video_cap = cv2.VideoCapture(mp4_file)
    i = 0
    out_name = mp4_file.split('.')[0]
    os.mkdir(os.path.join(OUT_DIR, out_name))
    logs(mp4_file)
    while True:
        ret, frame = video_cap.read()
        if ret == None:
            return
        sub = i // limit_frames
        if os.path.isdir(os.path.join(OUT_DIR, out_name, str(sub))):
            os.mkdir(os.path.join(OUT_DIR, out_name, str(sub)))
        cv2.imwrite(os.path.join(OUT_DIR, out_name, str(sub), f"frame_%06.jpg" %(i)))
        i += 1
        
        
if __name__ == "__main__":
    
    root = ""
    for path, subdir, files in os.walk(root):
        for name in files:
            n, ext = os.path.splitext(name)
            if ext in ['mkv', 'webm']:
                out_name = convert_to_mp4(os.path.join(path, name))
                if out_name == None:
                    continue
                extract_frame(out_name)
            else:
                extract_frame(os.path.join(path, name))
                
    

        
    
    
            
        

