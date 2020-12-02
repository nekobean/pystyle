import cv2
import numpy as np

# 前景画像を読み込む。
fg_img = cv2.imread("sample2.jpg")

# 背景画像を読み込む。
bg_img = cv2.imread("sample3.jpg")

# HSV に変換する。
hsv = cv2.cvtColor(fg_img, cv2.COLOR_BGR2HSV)

# 2値化する。
bin_img = cv2.inRange(hsv, (0, 10, 0), (255, 255, 255))

# 輪郭抽出する。
contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 面積が最大の輪郭を取得する
contour = max(contours, key=lambda x: cv2.contourArea(x))

# マスク画像を作成する。
mask = np.zeros_like(bin_img)
cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)

x, y = 10, 10  # 貼り付け位置

# 幅、高さは前景画像と背景画像の共通部分をとる
w = min(fg_img.shape[1], bg_img.shape[1] - x)
h = min(fg_img.shape[0], bg_img.shape[0] - y)

# 合成する領域
fg_roi = fg_img[:h, :w]
bg_roi = bg_img[y : y + h, x : x + w]

# 合成する。
bg_roi[:] = np.where(mask[:h, :w, np.newaxis] == 0, bg_roi, fg_roi)

# 保存する。
cv2.imwrite("output.jpg", bg_img)
