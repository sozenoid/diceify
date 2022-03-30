import cv2
import matplotlib.pyplot as plt
import numpy as np

def resize_pic(fname):
    n_dices = 10000
    dice_side = 0.8 # cm
    img = cv2.imread(fname)
    w, h, c = img.shape
    AS = float(w)/float(h)
    x = (n_dices/h**2/AS)**.5

    print(f'Image was be  {w} by {h} pixels')
    print(f'Image will be {int(AS*x*h)} by {int(x*h)} dices ({int(AS*x*h)* int(x*h)} dices)')
    print(f'Image will be {int(AS*x*h)*dice_side:2.2f} by {int(x*h)*dice_side:2.2f} cm x cm')

    img_small = cv2.resize(img, (int(x*h),int(x*h*AS)))


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

    # make the image as strings of numbers
    for i, row in enumerate(digits):
        print('{0:3d}'.format(i), '\t',''.join(list([str(x) for x in row])))

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
    small_img = resize_pic('./mamanetjean_cropped.png')
    down_sampled_img = colour_downsample(small_img)
    cv2.imwrite('./diced_mamanetjean_cropped.png', 255-down_sampled_img)
