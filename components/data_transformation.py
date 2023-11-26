import os 
import sys
import shutil
import pandas as pd
import numpy as np


class TransformData:

    def __init__(self,data) -> None:
        self.data = data


    def label_encoding(self,data):
        # train_df, val_df = self.transform_data()
        label = {'sofa':0,
                'chair':1,
                'pottedplant':2,
                'person':3,
                'sheep':4,
                'tvmonitor':5,
                'bottle':6,
                'dog':7,
                'car':8,
                'boat':9,
                'horse':10,
                'train':11,
                'cat':12,
                'diningtable':13,
                'bus':14,
                'motorbike':15,
                'aeroplane':16,
                'bird':17,
                'bicycle':18,
                'cow':19}
        return data

    def transform_data(self):
        df = pd.DataFrame(self.data,columns = ['filename', 'width', 'height', 'name','xmin','xmax','ymin','ymax'])
        df['x_center'] = ((df['xmin']+df['xmax'])/2)/df['width']
        df['y_center'] = ((df['ymin']+df['ymax'])/2)/df['height']
        df['Width'] = (df['xmax']-df['xmin'])/df['width']
        df['Height'] = (df['ymax']-df['ymin'])/df['height']

        cols = ['width','height','xmin','xmax','ymin','ymax']
        df = df.drop(cols, axis = 1)
        images = df['filename'].unique()
        img_df = pd.DataFrame(images, columns = ['filename'])
        img_train = tuple(img_df.sample(frac = 0.7)['filename'])  # shug=ffle and select 80% of images
        img_val = tuple(img_df.query(f'filename not in {img_train}')['filename'])

        train_df = df.query(f'filename in {img_train}')
        val_df = df.query(f'filename in {img_val}')

        train_df['id'] = train_df['name'].apply(self.label_encoding)
        val_df['id'] = val_df['name'].apply(self.label_encoding)

        return train_df, val_df

      
    def group_obj(self):

        train_df, val_df = self.transform_data()

        folder_train = "/Users/harshalrajput/Desktop/mlops_yolo/Dataset/train"
        folder_val = "/Users/harshalrajput/Desktop/mlops_yolo/Dataset/val"

        os.makedirs(folder_train, exist_ok=True)
        os.makedirs(folder_val, exist_ok=True)

        cols = ['filename','id' ,'x_center' ,'y_center' ,'Width' ,'Height']

        groupby_obj_train = train_df[cols].groupby('filename')
        groupby_obj_val = val_df[cols].groupby('filename')
        filename_series_train = pd.Series(groupby_obj_train.groups.keys())
        filename_series_val = pd.Series(groupby_obj_val.groups.keys())
        

        return groupby_obj_train, folder_train, groupby_obj_val,folder_val
    
    # move image and label into the respective folder

    def save_data(self, filename, folder_path, groupby_obj):

        # filename, folder_path, groupby_obj,_,_,_ = self.group_obj()
        
        #move images
        src = os.path.join('/Users/harshalrajput/Desktop/mlops_yolo/Dataset/raw_data/JPEGImages/'+filename)  # sourse of file
        des = os.path.join(folder_path,filename)  # destination for moving file
        
        shutil.move(src,des)  # move images to destination folder
        
        # save labels
        text_filename = os.path.join(folder_path, os.path.splitext(filename)[0]+'.txt')  # name for txt file
        #os.path.splitext remove extension from filename i.e jpg
        groupby_obj.get_group(filename).set_index('filename').to_csv(text_filename,sep = " ",index = False,header = False)


class InitiateTransformation:
    def __init__(self, data) -> None:
        self.data = data
        self.obj = TransformData(self.data)

    def initiate_transformation(self):
        groupby_obj_train, folder_train,groupby_obj_val,folder_val= self.obj.group_obj()
        filename_series_train = pd.Series(groupby_obj_train.groups.keys())
        filename_series_train.apply(self.obj.save_data, args = (folder_train, groupby_obj_train))

        filename_series_val = pd.Series(groupby_obj_val.groups.keys())
        filename_series_val.apply(self.obj.save_data, args = (folder_val, groupby_obj_val))

    
    
    
    


