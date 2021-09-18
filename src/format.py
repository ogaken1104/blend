import cv2
import os
import sys
import glob

project_name=sys.argv[1]
images=glob.glob(f'../data/{project_name}/output/*.jpg')

for fname in images:
    img=cv2.imread(fname)
    # 拡張子付き
    name=os.path.basename(fname)
    # 拡張子なし
    name=os.path.splitext(name)[0]
    # print(name)
    cv2.imwrite(fname[:-3]+'png',img)