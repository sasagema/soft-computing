import numpy as np
import cv2
from scipy.spatial import distance
import time
from scipy import ndimage
from vector import distance, pnt2line
from keras import models
import random
from collections import Counter
import os
import math

os.environ['THEANO_FLAGS'] = ""
video_name = 'video-';
video_extension = '.avi';
video_path = 'data/';
video_path2 = 'videos2/';
test = False
cc = -1
student = 'RA60-2014 Sasa Gemovic'
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'white':(255, 255, 255)}
red = colors['red']
frame_num_get_contour = 10

def main():
    prepare_out_file(student)
    model = models.load_model('model.h5')

    for i in range(0,10):
        reset_id()
        step = 1
        # pomocni brojac
        step_counter = 1
        # Konacna suma brojeva koji su presli preko linija
        sum = 0
        elements = []
        t = 0
        counter = 0
        times = []

        # color filter koristi se za uklanje suma i linija u boji sa frejma
        kernel = np.ones((3, 3), np.uint8)
        lower = np.array([150, 150, 150])
        upper = np.array([255, 255, 255])



        video_file = video_path + video_name + str(i) + video_extension
        print('Ucitava se %s' %(video_name + str(i)))


        capture = cv2.VideoCapture(video_file)
        # prvi frejm sa video snimka, koristi se kod pronalazenja linija
        first_frame = capture.read()[1]
        frame_number = 1

        # Pronalazenje linija sa prvog frejma Houhg metodom
        # lueLine, plava linija. sabiraju se brojevi koji prodju preko nje
        # greenLine, zelena linija, oduzimaju se brojevi koji predju preko nje

        blue_line, green_line = find_lines(first_frame)

        # ovde otvara video
        while(capture.isOpened()):

            # cita jedan frejm video snimka
            ret, frame = capture.read()
            if ret == False:
                break
            # step_counter += 1
            if step_counter != 1 or frame_number != 1:
                if step_counter <= step:
                    step_counter += 1
                    frame_number += 1
                    continue


            # print('Obradjujem frejm %s' % frame_number)
            frame_number += 1
            # ovde se radi obrada frejma

            start_time = time.time()

            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(frame, lower, upper) #izdvaja samo brojeve sa frejma
            #mask = cv2.dilate(mask, kernel)
            #mask = cv2.erode(mask, kernel)

            white_image = cv2.bitwise_and(frame, frame, mask=mask)
            white_image = cv2.cvtColor(white_image, cv2.COLOR_BGR2GRAY)
            binary_image = cv2.threshold(white_image, 1, 255, cv2.THRESH_BINARY)[1]
            binary_image = cv2.dilate(binary_image, kernel)

            contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,)

            #cv2.putText(imgC2, 'Br Cont: ' + str(len(contours)), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)
            frame = draw_line(frame, np.reshape(blue_line,(1,4)))
            frame = draw_line(frame, np.reshape(green_line,(1,4)))

            for contour in contours:

                x, y, w, h = cv2.boundingRect(contour)
                xc = x + w//2
                yc = y + h//2
                dxc = w
                dyc = h
                if (dxc > 11 or dyc > 11) and (dxc < 28 and dyc < 28):
                    # cv2.circle(frame, (xc, yc), 14, (25, 25, 255), 1)
                    cv2.rectangle(frame, (xc - dxc//2, yc-dyc//2), (xc + dxc//2, yc+dyc//2), (0, 255, 0), 1)

                    elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
                    # find in range
                    lst = in_range(16, elem, elements)
                    nn = len(lst)
                    if nn == 0:
                        elem['id'] = next_id()
                        elem['t'] = t
                        elem['pass_blue'] = False
                        elem['pass_green'] = False
                        elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                        elem['number'] = -1
                        elem['contours'] = []
                        contour_image = get_contour(white_image, elem['center'], elem['size'])
                        elem['contours'].append(contour_image)
                        elements.append(elem)
                    elif nn == 1:
                        lst[0]['center'] = elem['center']
                        lst[0]['t'] = t
                        lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                        contour_image = get_contour(white_image, elem['center'], elem['size'])
                        lst[0]['contour'] = contour_image
                        # Na odredjeni broj frejmova cuva sliku konture
                        if t % frame_num_get_contour == 0:
                            lst[0]['contours'].append(contour_image)
            for el in elements:
                tt = t - el['t']
                if tt < 3:
                    dist_blue_line, pnt_blue, r_b = pnt2line(el['center'], blue_line[0], blue_line[1])
                    dist_green_line, pnt_green, r_g = pnt2line(el['center'], green_line[0], green_line[1])
                    cv2.line(frame, pnt_blue, el['center'], (255, 0, 0), 1)
                    cv2.line(frame, pnt_green, el['center'], (0, 255, 0), 1)
                    if r_b > 0:
                        if dist_blue_line < 8:
                            if not el['pass_blue']:
                                el['pass_blue'] = True
                                counter += 1
                                cv2.circle(frame, el['center'], 11, colors['blue'], 1)
                                contours = el['contours']
                                number = number_winner(contours, 10, model)
                                el['number'] = number
                                sum += number
                    if r_g > 0:
                        if dist_green_line < 8:
                            if not el['pass_green']:
                                counter += 1
                                el['pass_green'] = True
                                if not el['pass_blue']:
                                    cv2.circle(frame, el['center'], 11, colors['green'], 1)
                                    contours = el['contours']
                                    number = number_winner(contours, 10, model)
                                    el['number'] = number
                                else:
                                    number = el['number']

                                sum -= number
                    id = el['id']
                    cv2.putText(frame, str(el['id']),
                                (el['center'][0] + 10, el['center'][1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'])
                    cv2.putText(frame, str(el['number']),
                                (el['center'][0] - 20, el['center'][1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['white'])

            elapsed_time = time.time() - start_time
            times.append(elapsed_time * 1000)
            cv2.putText(frame, 'Suma: ' + str(sum), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # print nr_objects
            t += 1
            # resized = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(video_name + str(i), frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()
        write_result(video_name + str(i) + video_extension, sum)
        print('Rezultat za ' + video_name + str(i) + video_extension + ' je ' + str(sum) + '\n\n')

def get_contour(image, center, size):
    """
    Iseca sliku konture sa prosledjene slike frejma video snimka.

    """
    x = center[0]
    y = center[1]
    w = size[0]
    h = size[1]
    #contour = image[y:y-h, x:x-w]
    #con = image[y - h // 2: y + h // 2, x - w // 2: x + w // 2]
    con = image[(y - h // 2) + 1: y + h // 2, (x - w // 2) + 1: x + w // 2]

    return con


def number_winner(contours, num_of_samples, model):
    """
    Vrsi predikciju prosledjenih cifara i vraca broj
    koji je najvise puta prediktovan.

    """
    contours_len = len(contours)
    if contours_len < num_of_samples:
        num_of_samples = contours_len
    random_contours = random.sample(contours, num_of_samples)

    numbers = []

    for contour in random_contours:
        for_predict = prepare_for_prediction(contour)
        number, probability = predict(for_predict, model)
        numbers.append(number)
        b = Counter(numbers)
        winner = b.most_common(1)[0][0]

    return winner

def fill(image):
    """
    Dopunjava sliku crnim pikseilma sa svih strana do dimenzija 28x28.
    :param image: Slika
    :return: Vraca sliku 28x28
    """
    if np.shape(image) != (28, 28):

        img = np.zeros((28,28))
        shape = image.shape
        cx = shape[1] // 2
        cy = shape[0] // 2
        h, w = image.shape[:2]
        dx = 14 - cx
        dy = 14 - cy
        x = (28 - np.shape(image)[0])//2
        y = (28 - np.shape(image)[1])//2
        img[dy:dy+h, dx:dx+w] = image
        return img

    else:
        return image


def prepare_for_prediction(image):
    """
    Prima sliku cifre za predikciju, kreira novu sliku dimenzija 28x28
    i originalnu sliku cifre postavlja u gornji levi ugao pa je zatim bluruje.
    :param image: Slika cifre
    :return: Slika dimezija 28x28, blurovana.
    """
    x, y = image.shape
    img = np.zeros((28, 28))
    img[:x, :y] = image
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    return blurred


def predict(image, model):
    """
    Sliku dim. 28x28 prebacije u vektor dimenzije koju zahteva model istrenirane
    neuronske mreze.
    :param image: 28x28 Slika
    :param model: Istreniran model neuronske mreze
    :return: Broj koji je prediktovala neuronska mreza i njegovu verovatnocu
    """
    flattened = image.flatten()
    for_predict = np.reshape(flattened, (1, 784))
    for_predict = for_predict.astype('float32')
    for_predict /= 255

    predicted = model.predict(for_predict)
    number = np.argmax(predicted)
    ind = np.unravel_index(np.argmax(predicted, axis=None), predicted.shape)
    probability = predicted[ind]

    return number, probability


def find_lines(frame):
    """
    Pronalazi plavu i zelenu liniju sa slike.

    """
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 50, 50])
    mask_blue = cv2.inRange(frame, lower_blue, upper_blue)
    res_blue = cv2.bitwise_and(frame, frame ,mask= mask_blue)

    lower_green = np.array([0, 100, 0])
    upper_green = np.array([50, 255, 50])
    mask_green = cv2.inRange(frame, lower_green, upper_green)
    res_green = cv2.bitwise_and(frame, frame, mask=mask_green)

    # uklanjanje suma pomocu erozije i dilacije
    kernel = np.ones((3, 3), np.uint8)

    mask_green = cv2.erode(mask_green, kernel, iterations=1)
    mask_green = cv2.dilate(mask_green, kernel, iterations=1)

    mask_blue = cv2.erode(mask_blue, kernel, iterations=1)
    mask_blue = cv2.dilate(mask_blue, kernel, iterations=1)

    blue_line_list, blue_line_array = find_longest_line(mask_blue)
    green_line_list, green_line_array = find_longest_line(mask_green)

    return blue_line_list, green_line_list


def draw_line(image, line):
    """
    Prima sliku i niz od 4 elementa.
    Vraca sliku sa iscrtanim linijama.
     """
    ret_val = image.copy()
    for x1, y1, x2, y2 in line:
        cv2.line(ret_val, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(ret_val, (x1, y1), 5, (0, 0, 255), 2)
        cv2.circle(ret_val, (x2, y2), 5, (0, 0, 255), 2)

    return ret_val


def find_longest_line(image):
    """
    Pronalazi najduzu liniju od svih linija koje su pronadjene
    Hough trasformacijom.
    """
    image_blur = cv2.GaussianBlur(image, (7, 7), 1)
    min_length = 300
    max_gap = 5
    lines = cv2.HoughLinesP(image_blur, 1, np.pi / 180, 50, min_length, max_gap)

    longest = lines[0]
    coord = longest[0]
    #dist = distance.euclidean((coord[0], coord[1]), (coord[2], coord[3]))
    dist = distance((coord[0], coord[1]), (coord[2], coord[3]))

    for line in lines:
        coord = line[0]
        A = (coord[0], coord[1])
        B = (coord[2], coord[3])
        #dst = distance.euclidean(A, B)
        dst = distance(A, B)

        if dst >= dist:
            dist = dst
            longest_line = line

    line = [(longest_line[0][0], longest_line[0][1]), (longest_line[0][2], longest_line[0][3])]

    return line, longest_line


def find_contours(image):
    """
    Pronalazi i oznacava konture sa slike
    """
    ret_val = 0
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (50, 100, 255), 3)

        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image, center, radius, (255, 0, 0), 2)
        cv2.drawContours(image, contours, -1, (0, 0, 255), 1)

    return ret_val

def check_contuour(line, contour):
    """
    Proerava poziciju konture u odnosu na liniju
    :return: True ako je kontura iza linije (presla liniju)
    :return: False ako nije presla liniju.
    """
    line_x1, line_y1 = line[0]
    line_x2, line_y2 = line[1]

    line_angle = math.degrees(math.atan(((line_y2 - line_y1) * (-1))/(line_x2 - line_x1)))

    contour_x, contour_y = contour['center']

    line_contour_angle = math.degrees(math.atan(((contour_y - line_y1) * (-1))/(contour_x - line_x1)))

    if line_contour_angle < line_angle:
        return True
    else:
        return False


def in_range(r, item, items):
    """
    Proverava vektorsku udaljenost konture sa ostalim konturama
    u odnosu prag r.
    """
    ret_val = []
    for obj in items:
        min_dist = distance(item['center'], obj['center'])
        if min_dist < r:
            ret_val.append(obj)
    return ret_val


def next_id():
    global cc
    cc += 1
    return cc


def reset_id():
    global cc
    cc = 0

def prepare_out_file(student):
    """
    Priprema izlazni fajl sa rezultatima.
    Brise stare rezultate.
    """
    file = open('out.txt', 'w')
    file.write(student + '\n')
    file.write('video   sum\n')
    file.close()


def write_result(video, result):
    """
    Upisuje rezultate u izlazni fajl.
    """
    file = open('out.txt', 'a')
    file.write(video + ' ')
    file.write(str(result) + '\n')
    file.close()
if __name__ == '__main__':
    main()
