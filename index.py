import cv2
import numpy as np
import os
import sys
from pdf2jpg import pdf2jpg
import img2pdf
from PIL import Image


source = '30458.pdf'
destination = './'
try:
    os.mkdir('out')
except:
    pass

# pdf2jpg.convert_pdf2jpg(source, destination, pages="ALL")

for i in range(len([name for name in os.listdir(source + '_dir/') if os.path.isfile(os.path.join(source + '_dir/', name))])):
    img = cv2.imread(source + '_dir/' + '{}_'.format(i) + source + '.jpg', 0)

    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.ones_like(img) * 255
    boxes = []

    for contour in contours:
        if (cv2.contourArea(contour) < 10000000) & (cv2.contourArea(contour) > 3000):
            hull = cv2.convexHull(contour)
            cv2.drawContours(mask, [hull], -1, 0, -1)
            x,y,w,h = cv2.boundingRect(contour)
            boxes.append((x,y,w,h))

    boxes = sorted(boxes, key=lambda box: box[0])

    mask = cv2.dilate(mask, np.ones((5,5),np.uint8))

    img[mask != 0] = 255

    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for n,box in enumerate(boxes):
        x,y,w,h = box
        cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(result, str(n),(x+5,y+17), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,0),2,cv2.LINE_AA)

    cv2.imwrite('out/res_{}.png'.format(i), result)

with open('output.pdf', 'wb') as f:
    f.write(img2pdf.convert([Image.open('out/'+i).filename for i in os.listdir('out/')]))
    f.close()
