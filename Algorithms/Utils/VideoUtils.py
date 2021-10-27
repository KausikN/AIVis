'''
Library for basic video functions
'''

# Imports
import os
import cv2
import subprocess
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Main Functions
def ReadImage(imgPath, imgSize=None, keepAspectRatio=False):
    I = cv2.imread(imgPath)
    if not imgSize == None:
        size_original = [I.shape[0], I.shape[1]]
        print(size_original)
        if keepAspectRatio:
            if imgSize[1] > imgSize[0]:
                imgSize = (size_original[0] * (imgSize[1] / size_original[1]), imgSize[1])
            elif imgSize[0] > imgSize[1]:
                imgSize = (imgSize[0], size_original[1] * (imgSize[0] / size_original[0]))
            else:
                if size_original[1] > size_original[0]:
                    imgSize = (size_original[0] * (imgSize[1] / size_original[1]), imgSize[1])
                else:
                    imgSize = (imgSize[0], size_original[1] * (imgSize[0] / size_original[0]))
            imgSize = (int(round(imgSize[1])), int(round(imgSize[0])))
        I = cv2.resize(I, imgSize)
    # I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    return I

def DisplayImage(I, title=''):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    plt.imshow(I, 'gray')
    plt.title(title)
    plt.show()

def SaveImage(I, path):
    cv2.imwrite(path, I)

def ReadVideo(path):
    cap = cv2.VideoCapture(path)
    return cap

def WebcamVideo():
    return cv2.VideoCapture(0)

def SaveFrames2Video(frames, pathOut, fps=20.0, size=(640, 480)):
    if os.path.splitext(pathOut)[-1] == '.gif':
        frames_images = [Image.fromarray(frame) for frame in frames]
        extraFrames = []
        if len(frames_images) > 1:
            extraFrames = frames_images[1:]
        frames_images[0].save(pathOut, save_all=True, append_images=extraFrames, format='GIF', loop=0)
    else:
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
        for frame in frames:
            out.write(frame)
        out.release()

def FixVideoFile(pathIn, pathOut):
    COMMAND_VIDEO_CONVERT = 'ffmpeg -i \"{path_in}\" -vcodec libx264 \"{path_out}\"'

    if os.path.exists(pathOut):
        os.remove(pathOut)

    convert_cmd = COMMAND_VIDEO_CONVERT.format(path_in=pathIn, path_out=pathOut)
    print("Running Conversion Command:")
    print(convert_cmd + "\n")
    ConvertOutput = subprocess.getoutput(convert_cmd)

# Driver Code
# Params

# # Params

# # RunCode