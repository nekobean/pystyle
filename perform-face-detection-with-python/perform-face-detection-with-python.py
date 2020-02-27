#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition
import matplotlib.pyplot as plt

# 保存されている人物の顔の画像を読み込む。
known_face_imgs = []
for path in ["known-face_01.png", "known-face_02.png", "known-face_03.png"]:
    img = face_recognition.load_image_file(path)
    known_face_imgs.append(img)

# 認証する人物の顔の画像を読み込む。
face_img_to_check = face_recognition.load_image_file("face_to_check.png")


# In[2]:


# 顔の画像から顔の領域を検出する。
known_face_locs = []
for img in known_face_imgs:
    loc = face_recognition.face_locations(img, model="cnn")
    known_face_locs.append(loc)

face_loc_to_check = face_recognition.face_locations(face_img_to_check, model="cnn")


# In[3]:


def draw_face_locations(img, locations):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_axis_off()
    for i, (top, right, bottom, left) in enumerate(locations):
        # 長方形を描画する。
        w, h = right - left, bottom - top
        ax.add_patch(plt.Rectangle((left, top), w, h, ec="r", lw=2, fill=None))
    plt.show()


for img, loc in zip(known_face_imgs, known_face_locs):
    draw_face_locations(img, loc)
    
draw_face_locations(face_img_to_check, face_loc_to_check)


# In[4]:


# 顔の領域から特徴量を抽出する。
known_face_encodings = []
for img, loc in zip(known_face_imgs, known_face_locs):
    (encoding,) = face_recognition.face_encodings(img, loc)
    known_face_encodings.append(encoding)

(face_encoding_to_check,) = face_recognition.face_encodings(
    face_img_to_check, face_loc_to_check
)


# In[5]:


matches = face_recognition.compare_faces(known_face_encodings, face_encoding_to_check)
print(matches)  # [True, False, False]


# In[6]:


dists = face_recognition.face_distance(known_face_encodings, face_encoding_to_check)
print(dists)


# In[ ]:




