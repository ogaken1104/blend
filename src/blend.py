import argparse
import numpy as np
import cv2
import os
from scipy.ndimage import rotate

parser=argparse.ArgumentParser()
parser.add_argument('project',help='projectの名前')
args=parser.parse_args()
# 画像を読み込むディレクトリ名
project_name = args.project
# 読み込みたいマーカーのidを指定
read_ids_front = [0, 1]
read_ids_back = [1, 0]
# 表画像の透過率
alpha = 0.5
# 必要に応じてmean_back(裏画像のマーカー中心座標)の値を補正する
back_correct = np.array([[0., 0.], [0., 0.]], np.float32)


def getMarkerMean(ids, corners, index):
    for i, id in enumerate(ids):
        # マーカーのインデックス検索
        if(id[0] == index):
            v = np.mean(corners[i][0], axis=0)  # マーカーの4隅の座標から中心の座標を取得する
            return [v[0], v[1]]
    return None


def getMarkerDistance(marker_mean):
    delta_xy = marker_mean[0]-marker_mean[1]
    return np.sqrt(np.sum(delta_xy*delta_xy))


def resize_and_show(image, size=(800, 600)):
    image_resized = cv2.resize(image, size)
    cv2.imshow('image_resized', image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rot(image, xy, angle):
    im_rot = rotate(image, angle)
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
                    -org[0]*np.sin(a) + org[1]*np.cos(a)])
    return im_rot, new+rot_center


dir_path = '../data/'+str(project_name)+'/output'
aruco = cv2.aruco
imgs = []
imgs.append(cv2.imread(os.path.join(dir_path, f'front.png')))
imgs.append(cv2.imread(os.path.join(dir_path, f'back.png')))
# imgs.append(cv2.imread(os.path.join(dir_path, f'{project_name}_front.jpg')))
# imgs.append(cv2.imread(os.path.join(dir_path, f'{project_name}_back.jpg')))

p_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)

# 表画像のマーカー中心読み込み
mean_front = np.zeros((2, 2))
corners, ids, rejectedImgPoints = aruco.detectMarkers(imgs[0], p_dict)
# print(f'表画像のids：{ids}')
for j, read_id in enumerate(read_ids_front):
    mean_front[j] = getMarkerMean(ids, corners, read_id)

# 裏画像のマーカー中心読み込み
mean_back = np.zeros((2, 2))
corners, ids, rejectedImgPoints = aruco.detectMarkers(imgs[1], p_dict)
# print(f'裏画像のids：{ids}')
for j, read_id in enumerate(read_ids_back):
    # print(j, read_id)
    mean_back[j] = getMarkerMean(ids, corners, read_id)


# マーカー中心間の距離を取得
distance_front = getMarkerDistance(mean_front)
distance_back = getMarkerDistance(mean_back)

# 裏画像のマーカー中心座標と画像配列を上下反転
mean_back[:, 1] = imgs[1].shape[0]-mean_back[:, 1]
# 裏画像のマーカー位置補正
mean_back += back_correct
# print(f'表画像のマーカー中心座標{mean_front}')
# print(f'裏画像のマーカー中心座標{mean_back}')
img1_reversed = cv2.flip(imgs[1], 0)

# 回転角の計算
a = mean_front[1]-mean_front[0]
b = mean_back[1]-mean_back[0]
b_minus_a = b-a
length_vec_a = np.linalg.norm(a)
length_vec_c = np.linalg.norm(b)
inner_product = np.inner(a, b)
cos = inner_product / (length_vec_a * length_vec_c)
# 角度（ラジアン）の計算
radian = np.arccos(cos)

# 裏画像のマーカー中心の中点
back_midpoint = np.average(mean_back, axis=0)

# 用いた回転行列の計算
mean_back_warped = np.zeros((2, 2, 2))
vector = []
error = []
matrix = []
i = 0
# 回転の向きを決定
# 要修正
for rad in (radian, -radian):
    mean_back_temp = np.ones((2, 3))
    mean_back_temp[:, :2] = mean_back-back_midpoint
    angle = rad*180.0/np.pi
    affine = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    matrix.append(affine)
    mean_back_warped[i, 0] = np.dot(matrix[i], mean_back_temp[0])
    mean_back_warped[i, 1] = np.dot(matrix[i], mean_back_temp[1])
    vector.append(mean_back_warped[i, 1]-mean_back_warped[i, 0])
    error.append(np.mean((vector[i]-a)**2))
    i += 1
# print(error)
# angle,-angleのときのベクトルをaと比較して向きを決定
if error[0] < error[1]:
    mean_back = mean_back_warped[0]
else:
    radian = -radian
    mean_back = mean_back_warped[1]

angle = radian*180.0/np.pi
back_warped, midpoint_warped = rot(img1_reversed, back_midpoint, angle)
# resize_and_show(back_warped)
# 変換したことによるmean_backの補正
mean_back += midpoint_warped

# 片方のマーカーを基準に平行移動
delta = mean_front[0]-mean_back[0]
src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
dst = src+delta.astype(np.float32)

# 裏画像の移動
af = cv2.getAffineTransform(src, dst)
# print(f'平行移動の回転行列：{af}')
back_warped = cv2.warpAffine(
    back_warped, af, (imgs[0].shape[1], imgs[0].shape[0])
)

synthesized_img = imgs[0]*alpha+back_warped*(1-alpha)
filename = os.path.join(dir_path, args.project+"_blended.png")
cv2.imwrite(filename, synthesized_img)

with open(os.path.join(dir_path, 'memo.txt'), 'w') as f:
    f.writelines([
        f'detected Marker coordinate\n',
        f'front\n',
        f'id={read_ids_front[0]}:{mean_front[0][0],mean_front[0][1]}\n',
        f'id={read_ids_front[1]}:{mean_front[1][0],mean_front[1][1]}\n',
        f'back\n',
        f'id={read_ids_back[0]}:{mean_back[0][0],mean_back[0][1]}\n',
        f'id={read_ids_back[1]}:{mean_back[1][0],mean_back[1][1]}\n'
        f'marker distance of front:{distance_front}\n',
        f'marker distance of back:{distance_back}\n'
    ]
    )
