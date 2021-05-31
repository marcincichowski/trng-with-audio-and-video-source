import numpy as np
import copy
from matplotlib import pyplot as plt
from math import log, e
import math
import cv2
from scipy.io import wavfile
from scipy import stats
from collections import Counter
#import random as rand

def compute_rank(rows, cols, matrix):
    minimum = min(rows, cols)
    i = 0
    while i < minimum - 1:
        if matrix[i][i] == 1:
            matrix = perform_row_operations(i, matrix, rows, forward_elimination=True)
        else:
            found = find_unit_element_swap(i, matrix, rows, forward_elimination=True)
            if found == 1:
                matrix = perform_row_operations(i, matrix, rows, forward_elimination=True)
        i += 1
    i = minimum - 1
    while i > 0:
        if matrix[i][i] == 1:
            matrix = perform_row_operations(i, matrix, rows, forward_elimination=False)
        else:
            if find_unit_element_swap(i, matrix, rows, forward_elimination=False) == 1:
                matrix = perform_row_operations(i, matrix, rows, forward_elimination=False)
        i -= 1
    return determine_rank(minimum, rows, cols, matrix)

def perform_row_operations(i, matrix, rows, forward_elimination):
    if forward_elimination:
        j = i + 1
        while j < rows:
            if matrix[j][i] == 1:
                matrix[j, :] = (matrix[j, :] + matrix[i, :]) % 2
            j += 1
    else:
        j = i - 1
        while j >= 0:
            if matrix[j][i] == 1:
                matrix[j, :] = (matrix[j, :] + matrix[i, :]) % 2
            j -= 1
    return matrix

def find_unit_element_swap(i, matrix, rows, forward_elimination):
    row_op = 0
    if forward_elimination:
        index = i + 1
        while index < rows and matrix[index][i] == 0:
            index += 1
        if index < rows:
            row_op = swap_rows(i, index, matrix)
    else:
        index = i - 1
        while index >= 0 and matrix[index][i] == 0:
            index -= 1
        if index >= 0:
            row_op = swap_rows(i, index, matrix)
    return row_op


def swap_rows(i, ix, matrix):
    temp = copy.copy(matrix[i, :])
    matrix[i, :] = matrix[ix, :]
    matrix[ix, :] = temp
    return 1


def determine_rank(minimum, rows, cols, matrix):
    rank = minimum
    i = 0
    while i < rows:
        all_zeros = 1
        for j in range(cols):
            if matrix[i][j] == 1:
                all_zeros = 0
        if all_zeros == 1:
            rank -= 1
        i += 1
    return rank


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


def save_result(number, files):
    byte_arr = [number["random_number"]]
    binary_format = bytearray(byte_arr)
    files["binary"].write(binary_format)


def random1(x, y, k, video, audio, n):


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
    results_bin = []
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

        for i in range(0, 32):
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
            results_bin.append(glue)
            string+=str(glue)

        results.append((int(string, 2)))
        R2 = R
        G2 = G
        B2 = B

    sound_x_plot = np.arange(sound_counter)
    with open("output.bin", "w") as f:
        for x in results_bin:
            f.write(str(x))
    print(results_bin)
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
#random(x_initial, y_initial, k_initial, "test.mp4", "test.wav",1000000)

def Phi(z):
    tmp = z/math.sqrt(2)
    tmp = 1+math.erf(tmp)
    return tmp/2

def park(filename):
    NUMBER_OF_TESTS = 100
    p_array = []
    z_array = []
    sigma = 21.9
    mu = 3523
    print("start parking lot test")
    with open(filename, "r") as f:
        for no in range(0, NUMBER_OF_TESTS):
            parking_lot_x_array = []
            parking_lot_y_array = []
            successfull_parking_counter = 0
            crash_counter = 0
            crash = False
            for x in range(0, 12000):
                random_32b_x = f.read(32)
                if len(random_32b_x) < 31:
                    f.close()
                    f = open(filename, "r")
                    random_32b_x = f.read(32)
                random_32b_y = f.read(32)
                if len(random_32b_y) < 31:
                    f.close()
                    f = open(filename, "r")
                    random_32b_y = f.read(32)
                random_32b_x = 100 * int(random_32b_x, 2) / 4294967296
                random_32b_y = 100 * int(random_32b_y, 2) / 4294967296
                for i in range(successfull_parking_counter):
                    if abs(parking_lot_x_array[i]-random_32b_x) <= 1 and abs(parking_lot_y_array[i]-random_32b_y) <= 1:
                        crash_counter += 1
                        crash = True
                        break
                if crash is False:
                    parking_lot_x_array.append(random_32b_x)
                    parking_lot_y_array.append(random_32b_y)
                    successfull_parking_counter += 1
                crash = False
            z = (successfull_parking_counter - mu) / sigma
            z_array.append(z)
            p = 1 - Phi(z)
            p_array.append(p)
            print("zaparkowano: "+str(successfull_parking_counter))
            print("wypadków: "+str(crash_counter))
        print(p_array)
        _, test = stats.kstest(p_array, 'uniform')
        print(test)
        plt.hist(z_array, bins=20,  weights=np.zeros_like(z_array) + 1. / len(z_array))
        plt.show()
        plt.hist(p_array, bins=20, weights=np.zeros_like(p_array) + 1. / len(p_array))
        plt.show()
    return

def connect_count(list_of_letters: list, word_length: int):
    words = list()
    for letter in range(0, 256000):
        word = "".join(list_of_letters[letter:letter + word_length])
        words.append(word)
    result = {}
    counter = Counter(words)
    for z in sorted(counter):
        result[z] = counter[z]
    return list(result.keys()), list(result.values())


def count_the_ones(filename):
    """
        0-2 - A
        3 - B
        4 - C
        5 - D
        6-8 - E
    """
    NUMBER_OF_TEST = 24
    probability_example = [37/256, 56/256, 70/256, 56/256, 37/256]
    letters_example = ['A', 'B', 'C', 'D', 'E']
    sum = 256000
    mean = 2500
    std = math.sqrt(5000)
    p_array = []
    print("start count the ones")
    with open(filename, "r") as f:
        for tests in range(0, NUMBER_OF_TEST):
            ones = 0
            letters = []
            for x in range(0, 256000+5):
                random_8b = f.read(8)
                if len(random_8b) < 7:
                    f.close()
                    f = open(filename, "r")
                    random_8b = f.read(8)
                for y in range(0, 8):
                    if random_8b[y] == "1":
                        ones += 1
                if ones < 3:
                    letters.append("A")
                else:
                    if ones == 3:
                        letters.append("B")
                    else:
                        if ones == 4:
                            letters.append("C")
                        else:
                            if ones == 5:
                                letters.append("D")
                            else:
                                letters.append("E")
                ones = 0
            letters_5, counts_5 = connect_count(letters, 5)
            letters_4, counts_4 = connect_count(letters, 4)

            probs_4 = {}
            for x in range(625):
                sum = 256000
                tmp = x
                iterals = []
                for y in range(4):
                    iterals.append(letters_example[tmp % 5])
                    sum *= probability_example[tmp % 5]
                    tmp = math.floor(tmp / 5)
                probs_4[''.join(iterals)] = sum

            probs_5 = {}
            for x in range(3125):
                sum = 256000
                tmp = x
                iterals = []
                for y in range(5):
                    iterals.append(letters_example[tmp % 5])
                    sum *= probability_example[tmp % 5]
                    tmp = math.floor(tmp / 5)
                probs_5[''.join(iterals)] = sum

            chi_4 = 0
            for x, y in zip(letters_4, counts_4):
                chi_4 += ((y - probs_4[x]) ** 2) / probs_4[x]

            chi_5 = 0
            for x, y in zip(letters_5, counts_5):
                chi_5 += ((y - probs_5[x]) ** 2) / probs_5[x]

            print("chi4 "+str(chi_4))
            print("chi5 "+str(chi_5))

            z = (chi_5 - chi_4 - mean)/std
            p = 1 - Phi(z)
            p_array.append(p)
        print("wyniki p dla wszystkich testów: "+ str(p_array))
        _, test = stats.kstest(p_array, "uniform")
        print("wynik kstest: "+str(test))
        x_plot_4 = np.arange(625)
        x_plot_5 = np.arange(3125)
        plt.plot(x_plot_4, counts_4)
        plt.title("rozkład wyrazów złożonych z 4 symboli")
        plt.show()
        plt.plot(x_plot_5, counts_5)
        plt.title("rozkład wyrazów złożonych z 5 symboli")
        plt.show()

        p_array, y = sorted(p_array), np.arange(len(p_array)) / len(p_array)
        ideal_x = [0.5] * len(p_array)
        ideal_y = np.arange(len(ideal_x)) / len(ideal_x)
        plt.plot(p_array, y)
        plt.plot(ideal_x, ideal_y, 'blue')
        plt.show()

        plt.hist(p_array, bins=len(p_array), weights=np.zeros_like(p_array) + 1/len(p_array))
        plt.show()
    return


def rank_matrix(matrix):
    n = len(matrix[0])
    rank = 0
    for col in range(n):
        j = 0
        rows = []
        while j < len(matrix):
            if matrix[j][col] == 1:
                rows += [j]
            j += 1
        if len(rows) >= 1:
            for c in range(1, len(rows)):
                for k in range(n):
                    matrix[rows[c]][k] = (matrix[rows[c]][k] + matrix[rows[0]][k]) % 2
            matrix.pop(rows[0])
            rank += 1
    for row in matrix:
        row_sum = 0
        for i in row:
            row_sum += int(i)
        if row_sum > 0:
            rank += 1
    return rank

def bin_matrix_rank_test(filename):
    p30 = [0.0052854502, 0.1283502644, 0.5775761902, 0.2887880952]
    number_of_matrices = 40000
    repeat_test = 24
    pvalue = []
    with open(filename, "r") as f:
        # powtarzanie testu
        for tests in range(0, repeat_test):
            print(tests)
            mat = []
            num_of = []
            num_of.append(0)
            num_of.append(0)
            num_of.append(0)
            num_of.append(0)
            for i in range(0, number_of_matrices):
                matrix = []
                for y in range(0, 32):
                    row = []
                    random_32b = f.read(32)
                    if len(random_32b) < 31:
                        f.close()
                        f = open(filename, "r")
                        random_32b = f.read(32)
                    # uzupełnianie macierzy kolejnymi bitami
                    for x in range(0, 32):
                        row.append(int(random_32b[x]))
                    matrix.append(row)
                # wyznaczenie rzędu macierzy
                matrix = np.array(matrix)
                tmp2 = compute_rank(32, 32, matrix)
                mat.append(tmp2)
            # liczenie ilości kolejnych rzędów
            for z in range(0, number_of_matrices):
                if mat[z] == 32:
                    num_of[3] += 1
                else:
                    if mat[z] == 31:
                        num_of[2] += 1
                    else:
                        if mat[z] == 30:
                            num_of[1] += 1
                            # 29, 28, 27 ,26...
                        else:
                            num_of[0] += 1
            # wyznaczanie wartosci oczekowianych
            p30_oczekiwana = []
            for i in range(0, 4):
                p30_oczekiwana.append(p30[i] * number_of_matrices)
            # test zgodności chi kwadrat
            sums = 0
            for i in range(0, 4):
                temp = num_of[i] - p30_oczekiwana[i]
                sums += pow((temp / p30_oczekiwana[i]), 2)
            # sum - wynik testu zgodności
            print("Wynik testu zgodności chi kwadrat: " + str(sums))
            print("wystepowanie macierzy rzedu 32: " + str(num_of[3]))
            print("wystepowanie macierzy rzedu 31: " + str(num_of[2]))
            print("wystepowanie macierzy rzedu 30: " + str(num_of[1]))
            print("wystepowanie macierzy rzedu < 30: " + str(num_of[0]))
            pvalue.append(sums)
        _, test = stats.kstest(pvalue, "uniform")

        print("wynik kstest: " + str(test))
        plt.hist(pvalue, bins=len(pvalue), weights=np.zeros_like(pvalue) + 1/len(pvalue))
        plt.show()


park("output.bin")
count_the_ones("output.bin")
bin_matrix_rank_test("output.bin")