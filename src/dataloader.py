#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import string
from IPython.display import display
from tqdm import tqdm
from PIL import Image

alphabet_string = string.ascii_lowercase
alphabet= list(alphabet_string)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Train set
"""
class Dgaze_DataSet():
   def __init__(self):
        samples_per_vid = 20
        self.image_list = []
        self.txt_list1 =  sorted(glob.glob("/content/drive/MyDrive/DGAZE/road_view/*.txt"))
        self.labels=[]
        for i in range(len(self.txt_list1)):
          df = pd.read_csv(self.txt_list1[i], header = None)
          for j in range(samples_per_vid):

            label_arr = np.array([(df.iloc[j][1]+df.iloc[j][3])/2, (df.iloc[j][2]+df.iloc[j][4])/2])
            lab_tensor = torch.from_numpy(label_arr).to(device=device)
            self.labels = self.labels + [lab_tensor]
        self.labels = self.labels*11
        for i in range(6,24):
          # 7, 14, 17, 18, 20, 22, 13
          if(i not in (7, 13, 14, 17, 18, 20, 22)):
            file_name = "/content/drive/MyDrive/DGAZE/full_list"+str(i)
            open_file = open(file_name, "rb")
            list_f = pickle.load(open_file)
            for i in list_f:
                  min_ele = torch.min(i[0])
                  i[0] -= min_ele
                  i[0] /= torch.max(i[0])
                  i[0] = i[0].to(device=device)
                  min_ele1 = torch.min(i[1])
                  i[1] -= min_ele1
                  i[1] /= torch.max(i[1])
                  i[1] = i[1].to(device=device)
            print(i, len(list_f))
            self.image_list=self.image_list+list_f
            open_file.close()

   def __getitem__(self,index):
        return self.image_list[index], self.labels[index]

   def get_list(self):
     return self.txt_list1

   def __len__(self):
        return len(self.labels)


"""
Validation set
"""
class Dgaze_ValSet():
   def __init__(self):
        samples_per_vid = 20
        no_of_drivers = 1
        self.image_list = []
        self.txt_list1 =  sorted(glob.glob("/content/drive/MyDrive/DGAZE/road_view/*.txt"))
        self.labels=[]
        for i in range(len(self.txt_list1)):
          df = pd.read_csv(self.txt_list1[i], header = None)
          for j in range(samples_per_vid):

            label_arr = np.array([(df.iloc[j][1]+df.iloc[j][3])/2, (df.iloc[j][2]+df.iloc[j][4])/2])
            lab_tensor = torch.from_numpy(label_arr).to(device=device)
            self.labels = self.labels + [lab_tensor]
        self.labels = self.labels*no_of_drivers
        for i in range(6,24):
          # 7, 14, 17, 18, 20, 22, 13
          if(i in (22,25)):
            file_name = "/content/drive/MyDrive/DGAZE/full_list"+str(i)
            open_file = open(file_name, "rb")
            list_f = pickle.load(open_file)
            for i in list_f:
                  min_ele = torch.min(i[0])
                  i[0] -= min_ele
                  i[0] /= torch.max(i[0])
                  i[0] = i[0].to(device=device)
                  min_ele1 = torch.min(i[1])
                  i[1] -= min_ele1
                  i[1] /= torch.max(i[1])
                  i[1] = i[1].to(device=device)
            print(i, len(list_f))
            self.image_list=self.image_list+list_f
            open_file.close()

   def __getitem__(self,index):
        return self.image_list[index], self.labels[index]

   def get_list(self):
     return self.txt_list1

   def __len__(self):
        return len(self.labels)


"""
Test set
"""
class Dgaze_TestSet():
   def __init__(self):
        samples_per_vid = 20
        no_of_drivers = 1
        self.image_list = []
        self.txt_list1 =  sorted(glob.glob("/content/drive/MyDrive/DGAZE/road_view/*.txt"))
        self.labels=[]
        for i in range(len(self.txt_list1)):
          df = pd.read_csv(self.txt_list1[i], header = None)
          for j in range(samples_per_vid):
            if (self.txt_list1[i]=='/content/drive/MyDrive/DGAZE/road_view/trip79_out.txt'):
              if(j in (18,19)):
                break
            label_arr = np.array([(df.iloc[j][1]+df.iloc[j][3])/2, (df.iloc[j][2]+df.iloc[j][4])/2])
            lab_tensor = torch.from_numpy(label_arr).to(device=device)
            self.labels = self.labels + [lab_tensor]
        self.labels = self.labels*no_of_drivers
        for i in range(6,24):
          # 7, 14, 17, 18, 20, 22, 13
          if(i ==17):
            file_name = "/content/drive/MyDrive/DGAZE/full_list"+str(i)
            open_file = open(file_name, "rb")
            list_f = pickle.load(open_file)
            for i in list_f:
                  min_ele = torch.min(i[0])
                  i[0] -= min_ele
                  i[0] /= torch.max(i[0])
                  i[0] = i[0].to(device=device)
                  min_ele1 = torch.min(i[1])
                  i[1] -= min_ele1
                  i[1] /= torch.max(i[1])
                  i[1] = i[1].to(device=device)
            print(i, len(list_f))
            self.image_list=self.image_list+list_f
            open_file.close()

   def __getitem__(self,index):
        return self.image_list[index], self.labels[index]

   def get_list(self):
     return self.txt_list1

   def __len__(self):
        return len(self.labels)
