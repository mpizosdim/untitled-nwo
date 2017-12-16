import numpy as np
import random
import cv2
import matplotlib.pyplot as plt


def get_imgs(df, reshape=True):
    imgs = []
    for i, row in df.iterrows():
        if reshape:
            band_1 = np.array(row['band_1']).reshape(75, 75)
            band_2 = np.array(row['band_2']).reshape(75, 75)
        else:
            band_1 = np.array(row['band_1'])
            band_2 = np.array(row['band_2'])
        band_3 = band_1 + band_2  # plus since log(x*y) = log(x) + log(y)
        imgs.append(np.dstack((band_1, band_2, band_3)))
    return np.array(imgs)


def rescale_img(df):
    imgs = []
    for img in df:
        band_1 = img[:, :, 0]
        band_2 = img[:, :, 1]
        band_3 = img[:, :, 2]
        band_1 = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        band_2 = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        band_3 = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
        imgs.append(np.dstack((band_1, band_2, band_3)))
    return np.array(imgs)


def _noisy(image, mean=1, var=0.1):
    row, col, ch = image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    return image + gauss


def _clone(img, case='duplicate'):
    if case == 'duplicate':
        new_img = np.concatenate((img, img),axis=1)
    elif case == 'duplicate_noise':
        variance = np.var(img) / 1000
        noisy_img = _noisy(img, random.uniform(-5, 5), variance)
        new_img = np.concatenate((img, noisy_img), axis=1)
    elif case == '4_noise2':
        variance = np.var(img) / 1000
        concat_img = np.concatenate((img, img),axis=1)
        noisy_img = _noisy(concat_img, random.uniform(-5, 5), variance)
        new_img = np.concatenate((concat_img, noisy_img), axis=0)
    return cv2.resize(new_img, dsize=(75, 75))


def clone_all(imgs, y):
    new_imgs = []
    labels = []
    for img, is_iceberg in zip(imgs, y):
        dupli = _clone(img, 'duplicate')
        dupli_noise = _clone(img, 'duplicate_noise')
        dupli2_noise = _clone(img, '4_noise2')
        new_imgs.extend((dupli, dupli_noise, dupli2_noise))
        labels.extend((is_iceberg * np.ones(3)))
    return np.concatenate((imgs, np.array(new_imgs))), np.concatenate((np.array(y), np.array(labels)))


def generate_new_data(x, y, times, generator):
    x_new = []
    y_new = []
    count = 0
    for x_, y_ in generator.flow(x, y, batch_size=len(x)):
        x_new.extend(x_)
        y_new.extend(y_)
        count += 1
        if count == times:
            break
    x_all = np.concatenate((x, x_new))
    y_all = np.concatenate((y, y_new))
    return x_all, y_all


def show_image(image, level=0):
    fig = plt.figure(1, figsize=(7, 7))
    img = image[:, :, level]
    plt.imshow(img)
    plt.show()
    pass

