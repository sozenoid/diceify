import cv2
import matplotlib.pyplot as plt
import numpy as np

def resize_pic(fname):
    img = cv2.imread(fname)
    img_small = cv2.resize(img, (100,100))
    grayImage = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    return grayImage

def colour_downsample(img):
    dice_dic = get_dice_pixelated()
    bins =  np.linspace(-1, 255, 7)[::-1]
    digits = np.digitize(img, bins, right=True)
    # down_sampled_img = np.array([[int(bins[x]) for x in y] for y in digits])
    down_sampled_img = np.array([[dice_dic[x][0] for x in y]+[dice_dic[x][1] for x in y]+[dice_dic[x][2] for x in y] for y in digits])
    down_sampled_img = np.reshape(down_sampled_img,(300,300))

    for row in digits:
        print(row)
    # print(down_sampled_img)
    plt.imshow(down_sampled_img, cmap='binary')
    plt.show()

def get_dice_pixelated():
    one =  np.array([[0, 0, 0],
                     [0, 1 ,0],
                     [0, 0, 0]])*255
    two =  np.array([[0, 0, 1],
                     [0, 0, 0],
                     [1, 0, 0]])*255
    three = np.array([[0, 0, 1],
                     [0, 1 ,0],
                     [1, 0, 0]])*255
    four = np.array([[1, 0, 1],
                     [0, 0 ,0],
                     [1, 0, 1]])*255
    five = np.array([[1, 0, 1],
                     [0, 1 ,0],
                     [1, 0, 1]])*255
    six =  np.array([[1, 0, 1],
                     [1, 0 ,1],
                     [1, 0, 1]])*255
    return {1:one,2:two,3:three,4:four,5:five,6:six}

if __name__ == "__main__":
    small_img = resize_pic('./maman_et_jean_square.jpeg')
    colour_downsample(small_img)
