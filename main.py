import numpy as np
from matplotlib import pyplot as plt
from moviepy.editor import *
from math import log, e
import cv2
from scipy.io import wavfile


def entropy(labels, base=2):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

def color(x, y, frame):
    return (frame[y, x, 1] << 16) + (frame[y, x, 2] << 8) + (frame[y, x, 0])

x_initial = 170
y_initial = 100
k_initial = 500

def random(x, y, k, video, audio, n):
    arr_b = []
    arr_r = []
    arr_g = []
    arr_sound = []
    counter = 0
    sound_counter = 0
    cap1 = cv2.VideoCapture(video)
    samplerate1, data1 = wavfile.read(audio)
    while counter != 1000000:
        counter = counter + 1
        res, frame = cap1.read()
        if frame is not None:
            frameHeight, frameWidth, channels = frame.shape
            arr_b.append(frame[x, y, 0])  # B Channel Value
            arr_r.append(frame[x, y, 1])  # R Channel Value
            arr_g.append(frame[x, y, 2])  # G Channel Value
        else:
            break
    arr_b_np = np.array(arr_b)
    arr_r_np = np.array(arr_r)
    arr_g_np = np.array(arr_g)

    x_plot = np.arange(counter - 1)
    plt.plot(x_plot, arr_b_np, 'b')
    plt.plot(x_plot, arr_r_np, 'r')
    plt.plot(x_plot, arr_g_np, 'g')
    plt.show()


    plt.hist(arr_b_np, bins=255)
    plt.show()
    plt.hist(arr_r_np, bins=255)
    plt.show()
    plt.hist(arr_g_np, bins=255)
    plt.show()

    results = []
    R1 = 0
    G1 = 0
    B1 = 0
    R2 = 0
    G2 = 0
    B2 = 0
    watchdog = 0
    th = 100
    chanel = 0
    vt = 3
    runcnt = 0
    cap = cv2.VideoCapture(video)
    samplerate, data = wavfile.read(audio)
    res, frame = cap.read()
    frameHeight, frameWidth, channels = frame.shape
    color_i = ((color(x - 1, y - 1, frame) + color(x - 1, y, frame) + color(x - 1, y + 1, frame)) / 9) + (
            (color(x, y - 1, frame) + color(x, y, frame) + color(x, y + 1, frame)) / 9) + (
                      (color(x + 1, y - 1, frame) + color(x + 1, y, frame) + color(x + 1, y + 1, frame)) / 9)
    x = color_i % (frameWidth/2) + frameWidth/4
    y = color_i % (frameHeight /2) + frameHeight / 4
    j = 0
    l = 0
    channel = 0
    for h in range(0, n):  # numbers generated
        R = frame[int(y), int(x), 1]
        G = frame[int(y), int(x), 2]
        B = frame[int(y), int(x), 0]
        while (((R - R1) ** 2) + ((G - G1) ** 2) + ((B - B1) ** 2)) < vt:
            R = frame[int(y), int(x), 1]
            G = frame[int(y), int(x), 2]
            B = frame[int(y), int(x), 0]
            if watchdog > th:
                res, frame = cap.read()
                watchdog = 0
            x = (x + (R ^ G) + 1) % frameWidth
            y = (y + (R ^ B) + 1) % frameHeight
            watchdog = watchdog + 1
        random_bit = []

        for i in range(0, 8):
            byte_sound = []
            if (l+1)*k > 6045696:
                channel = channel+1
                if channel >= 6:
                    channel = 0
                l = 0
            for iter in range(l * k, (l + 1) * k):
                byte_sound.append(data[iter][chanel])
                if sound_counter < 100000:
                    arr_sound.append(data[iter][chanel])
                    sound_counter = sound_counter + 1

            SN1_index = 10 + (R * i + (G << 2) + B + runcnt) % (k/2)
            SN2_index = 15 + (R * i + (G << 3) + B + runcnt) % (k/2)
            SN3_index = 20 + (R * i + (G << 4) + B + runcnt) % (k/2)
            SN4_index = 5 + (R * i + (G << 1) + B + runcnt) % (k/2)
            SN5_index = 25 + (R * i + (G << 5) + B + runcnt) % (k/2)

            SN1 = byte_sound[int(SN1_index)]
            SN2 = byte_sound[int(SN2_index)]
            SN3 = byte_sound[int(SN3_index)]
            SN4 = byte_sound[int(SN4_index)]
            SN5 = byte_sound[int(SN5_index)]

            random_bit.append(1 & (R ^ G ^ B ^ R1 ^ G1 ^ B1 ^ R2 ^ G2 ^ B2 ^ int(SN1) ^ int(SN2) ^ int(SN3) ^ int(SN4) ^ int(SN5)))
            R1 = R
            G1 = G
            B1 = B
            x_old = x
            y_old = y
            x = (((R ^ int(x_old)) << 4) ^ (G ^ int(y_old))) % frameWidth
            y = (((G ^ int(x_old)) << 4) ^ (B ^ int(y_old))) % frameHeight
            j = j + 1
            l = l + 1
            if j % 100000 == 0:
                runcnt = runcnt+1
        string = ""
        for glue in random_bit:
            string+=str(glue)
        results.append(int(string, 2))
        R2 = R
        G2 = G
        B2 = B
    sound_x_plot = np.arange(sound_counter)

    plt.plot(sound_x_plot, arr_sound)
    plt.show()
    plt.hist(arr_sound, bins=256)
    plt.show()

    plt.hist(results, bins=256)
    plt.show()
    print(entropy(arr_b_np))
    print(entropy(arr_g_np))
    print(entropy(arr_r_np))
    print(entropy(results))
    return
random(x_initial, y_initial, k_initial, "test.mp4", "test.wav", 100000)
