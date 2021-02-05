import cv2 
import os
import tqdm

if __name__ == '__main__':
    
    root_path = 'RWF-2000'
    sub_folders = ['train', 'val']
    classes_name = ['Fight', 'NonFight']
    dst_foler = 'RWF_2000/frames'
    
    for fl in sub_folders:
        for cls in classes_name:
            lst_video = os.listdir(os.path.join(root_path, fl, cls))
            for vi in tqdm.tqdm(lst_video):
                
                file_path = os.path.join(root_path, fl, cls, vi)
                assert os.path.isfile(file_path), "ERROR"
                
                cap = cv2.VideoCapture(file_path)
                len_frames = int(cap.get(7))   
                if not os.path.exists(os.path.join(dst_foler, vi.replace('.avi', ''))):
                    os.mkdir(os.path.join(dst_foler, vi.replace('.avi', '')))         
                try: 
                    for i in range(len_frames - 1):
                        if not os.path.exists(os.path.join(dst_foler, vi.replace('.avi', ''), "image_%05d.jpg" %(i + 1))):
                            _, frame = cap.read()
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(os.path.join(dst_foler, vi.replace('.avi', ''), "image_%05d.jpg" %(i + 1)), frame)
                except Exception as e:
                    print(e)
                    print("ERROR", vi)      
                
                finally:
                    cap.release()
                    if not os.path.exists(os.path.join(dst_foler, vi.replace('.avi', ''), 'n_frames')):
                        with open(os.path.join(dst_foler, vi.replace('.avi', ''), 'n_frames'), 'w') as fw:
                            fw.write(str(len_frames - 1))                      
                
                
                