#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition
import matplotlib.pyplot as plt

# 画像を読み込む。
img = face_recognition.load_image_file("sample.png")


# In[2]:


# 画像から顔の領域を検出する。
face_locs = face_recognition.face_locations(img, model="cnn")


# In[3]:


from PIL import ImageDraw, Image
from IPython.display import display

def draw_faces(img, locs):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img, mode="RGBA")

    for top, right, bottom, left in locs:
        draw.rectangle((left, top, right, bottom), outline="lime", width=2)

    display(img)


draw_faces(img, face_locs)


# In[4]:


import face_recognition
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# 画像を読み込む。
img = face_recognition.load_image_file("sample2.png")

# HOG 特徴量を使った顔検出
locs = face_recognition.face_locations(img, model="hog")
draw_faces(img, locs)

# CNN を使った顔検出
locs = face_recognition.face_locations(img, model="cnn")
draw_faces(img, locs)


# In[5]:


from pprint import pprint

import face_recognition
import matplotlib.pyplot as plt
import numpy as np

# 画像を読み込む。
img = face_recognition.load_image_file("sample3.png")

# 画像から顔の領域を検出する。
face_locs = face_recognition.face_locations(img, model="cnn")

# 顔の各部位を検出する。
facial_landmarks = face_recognition.face_landmarks(img, face_locs)
#pprint(facial_landmarks)

# 日本語訳
jp_names = {'nose_bridge': '鼻筋',
            'nose_tip': '鼻先',
            'top_lip': '上唇',
            'bottom_lip': '下唇',
            'left_eye': '左目',
            'right_eye': '左目',
            'left_eyebrow': '左眉毛',
            'right_eyebrow': '右眉毛',
            'chin': '下顎'}

# 可視化する。
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img)
ax.set_axis_off()
for face in facial_landmarks:
    for name, points in face.items():
        points = np.array(points)
        ax.plot(points[:, 0], points[:, 1], 'o-', ms=3, label=jp_names[name])
ax.legend(fontsize=14)

plt.show()


# In[ ]:




