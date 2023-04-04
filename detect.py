import cv2
import os
from BalloonDetector import detectFromPaths


def main():
    # 漫画の画像ファイルがあるディレクトリ
    input_dir = "./input_images"
    # フキダシを囲む矩形を描画した画像を保存するディレクトリ
    output_dir = "./output_images"

    # ディレクトリ内の画像ファイルのパスを取得
    input_paths = [os.path.join(input_dir, filename)
                   for filename in os.listdir(input_dir)]

    # フキダシを検出して矩形で囲み、画像を保存
    images, locations, new_paths = detectFromPaths(
        input_paths)
    for i in range(len(images)):
        image = images[i]
        location = locations[i]
        _new_path = new_paths[i].replace('.jpg', '')
        new_path = f'{_new_path}_{str(i)}.jpg'
        cv2.imwrite(os.path.join(
            output_dir, os.path.basename(new_path)), image)


if __name__ == '__main__':
    main()
