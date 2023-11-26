import os 
import sys
import shutil

import xml.etree.ElementTree as ET
from functools import reduce
import pandas as pd
import numpy as np

from logger import logging
from exception import CustomException
from components.data_transformation import InitiateTransformation


class Dataload:

    def load_data(self ):
        raw_xml = "/Users/harshalrajput/Desktop/mlops_yolo/Dataset/raw_data/Annotations"
        raw_image = "/Users/harshalrajput/Desktop/mlops_yolo/Dataset/raw_data/JPEGImages"

        xml_file = []
        for i in os.listdir(raw_xml):
            xml_file.append(raw_xml+ '/'+i)
        return xml_file
    
class ExtractData:

    def __init__(self) -> None:
        obj = Dataload()
        self.filename = obj.load_data()

    def extract(self, filename):
    
        tree = ET.parse(filename)

        root = tree.getroot()
        

        img_name = root.find('filename').text
        img_name

        width = float(root.find('size').find('width').text)
        height = float(root.find('size').find('height').text)
        objs = root.findall('object')

        parser = []
        
        for obj in objs:
            name = obj.find('name').text
            bndbox  = obj.find('bndbox')
            xmin =  float(bndbox.find('xmin').text)
            xmax =  float(bndbox.find('xmax').text)
            ymin =  float(bndbox.find('ymin').text)
            ymax =  float(bndbox.find('ymax').text)

            parser.append([img_name, width, height, name,xmin,xmax,ymin,ymax])


            return parser

class DataIngest:
    def __init__(self) -> None:
        obj1 = ExtractData()
        obj2 = Dataload()
        self.parser = obj1
        self.xml = obj2.load_data()

    def ingest_data(self):
        parser_all = list(map(self.parser.extract, self.xml))
        data = reduce(lambda x,y: x+y , parser_all)  # for flattening parser_all

        # df = pd.DataFrame(data,columns = ['filename', 'width', 'height', 'name','xmin','xmax','ymin','ymax'])
        # os.makedirs(os.path.join(os.getcwd(),"csv"))
        # df.to_csv(os.path.join(os.getcwd(),"csv/data.csv"))

        return data
    

if __name__ ==  "__main__":
    obj_ingt = DataIngest()
    data = obj_ingt.ingest_data()
    obj_transform = InitiateTransformation(data)
    obj_transform.initiate_transformation()




