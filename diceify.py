import cv2
import matplotlib.pyplot as plt
import numpy as np

def resize_pic(fname):
    n_dices = 200
    img = cv2.imread(fname)
    img_small = cv2.resize(img, (n_dices,n_dices))
    grayImage = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    return grayImage

def colour_downsample(img):
    dice_dic = get_dice_pixelated()
    bins =  np.linspace(-1, 255, 7)[::-1]
    digits = np.digitize(img, bins, right=True)

    dice_size = dice_dic[1].shape

    x_shape_new, y_shape_new = digits.shape[0]*dice_size[0], digits.shape[1]*dice_size[1]


    down_sampled_img = [np.concatenate([dice_dic[x] for x in y], axis=1) for y in digits]
    down_sampled_img = np.reshape(down_sampled_img,(x_shape_new, y_shape_new))


    plt.imshow(down_sampled_img, cmap='binary')
    plt.show()
    return down_sampled_img

def get_dice_pixelated(dice_size=5):
    if dice_size==3:
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
    elif dice_size==5:
        one =  np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0 ,0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])*255
        two =  np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0 ,1, 0],
                         [0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0]])*255
        three = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0 ,1, 0],
                         [0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0]])*255
        four = np.array([[0, 0, 0, 0, 0],
                         [0, 1, 0 ,1, 0],
                         [0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 0],
                         [0, 0, 0, 0, 0]])*255
        five = np.array([[0, 0, 0, 0, 0],
                         [0, 1, 0 ,1, 0],
                         [0, 0, 1, 0, 0],
                         [0, 1, 0, 1, 0],
                         [0, 0, 0, 0, 0]])*255
        six =  np.array([[0, 0, 0, 0, 0],
                         [0, 1, 0 ,1, 0],
                         [0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0],
                         [0, 0, 0, 0, 0]])*255
    else:
        print('Dices not defined')
        return
    return {1:one,2:two,3:three,4:four,5:five,6:six}

if __name__ == "__main__":
    small_img = resize_pic('./domestic-dog_thumb_square.jpg')
    down_sampled_img = colour_downsample(small_img)
    cv2.imwrite('./diced_domestic-dog_thumb_square.jpg', 255-down_sampled_img)
