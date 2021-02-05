import os

if __name__ == '__main__':
    
    root_path = 'RWF-2000'
    sub_folders = ['train', 'val']
    class_name = ['Fight', 'NonFight']
    
    for fl in sub_folders:
        for cls in class_name:
            
            lst_video = os.listdir(os.path.join(root_path, fl, cls))
            
            for i, vi in enumerate(lst_video):
                src_path = os.path.join(root_path, fl, cls, vi)
                dst_path = os.path.join(root_path, fl, cls, f"{fl}_{cls}_{str(i)}.avi")
                os.rename(src_path, dst_path)
                