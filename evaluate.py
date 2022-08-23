from os import device_encoding
import sys 
import torch
import imageio
import torch.nn as nn
import cv2
from tqdm import tqdm

import numpy as np
import os
from sklearn.preprocessing import normalize
import argparse
parser = argparse.ArgumentParser(description='OCFR script')

parser.add_argument('--device', default='gpu', help='gpu id')
parser.add_argument('--model_path', default='model_path', help='path to pretrained model')
parser.add_argument('--image_list', type=str, default='',help='pairs file')

parser.add_argument('--list_pairs', type=str, default='',help='pairs file')
parser.add_argument('--data_path', type=str, default='',help='root path to data')

parser.add_argument('--save_path', type=str, default='',help='root path to data')





class FaceModel():
    def __init__(self, model_path, save_path,image_list,data_path,ctx_id):
        self.gpu_id=ctx_id
        self.model=self._get_model(model_path)
        self.save_path=save_path
        if not(os.path.isdir(self.save_path)):
            os.makedirs(save_path)
        self.image_list=image_list
        self.data_path=data_path
    def _get_model(self, model_path):
        pass

    def _getFeatureBlob(self,input_blob):
        pass
    def read(self,image_path):
        pass

    def save(self,features,image_path_list,alignment_results):
        # Save embedding as numpy to disk in save_path folder
        for i in tqdm(range(len(features))):
            filename = str(str(image_path_list[i]).split("/")[-1].split(".")[0])
            np.save(os.path.join(self.save_path, filename), features[i])
        np.save(os.path.join(self.save_path,"alignment_results.txt"),np.asarray(alignment_results))
    def process(self,image,bbox):
        pass

    def distance(self,embedding1, embedding2):
        pass

    def save_score(self,score):
        pass

    def comparison(self,list_pairs):
        with open(list_pairs, "r") as f:
            for line in f:
                emb1=np.load(os.path.join(self.save_path,line.split()[0]+".npy"))
                emb2=np.load(os.path.join(self.save_path,line.split()[1]+".npy"))
                score=self.distance(emb1,emb2)
                self.save_score(score)

    def read_img_path_list(self):
        with open(self.image_list, "r") as f:
            lines = f.readlines()
            file_path = [os.path.join(self.data_path, line.rstrip().split()[0]) for line in lines]
            bbx = [line.rstrip().split()[1:] for line in lines]
        return file_path ,bbx


    def get_batch_feature(self, image_path_list, bbx, batch_size=64, flip=0):
        
        count = 0
        num_batch =  int(len(image_path_list) / batch_size)
        features = []
        alignment_results = []
        for i in range(0, len(image_path_list), batch_size):

            if count < num_batch:
                tmp_list = image_path_list[i : i+batch_size]
                tmp_list_bbx = bbx[i : i+batch_size]

            else:
                tmp_list = image_path_list[i :]
                tmp_list_bbx = bbx[i :]

            count += 1

            images = []
            for i  in range(len(tmp_list)):
                image_path=tmp_list[i]
                bbox=tmp_list_bbx[i]
                image=self.read(image_path)
                image,alignment_result=self.process(image,bbox)
                alignment_results.append(alignment_result)
                images.append(image)
            input_blob = np.array(images)

            emb = self._getFeatureBlob(input_blob)
            features.append(emb)
        features = np.vstack(features)
        self.save(features,image_path_list,alignment_results)
        return

def main(args):
    model=FaceModel(args.model_path,args.save_path,args.image_list,args.data_path,args.device)
    file_path, bbx= model.read_img_path_list()
    model.get_batch_feature(file_path, bbx)
    model.comparison(args.list_pairs)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

















