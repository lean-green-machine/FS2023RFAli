import pytesseract as tess
tess.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR/tesseract.exe"
import PIL.Image
import cv2
import numpy as np

#contrasting
Image = cv2.imread("test.png")
Image2 = np.array(Image, copy=True)
white_px = np.asarray([227, 198, 198])
black_px = np.asarray([0  , 0  , 0  ])

row, col, _ = Image.shape



for r in range(row):
    for c in range(col):
        px = Image[r][c]
        if all(px <= white_px):
            Image2[r][c] = black_px

cv2.imwrite("testnew.png", Image2)


myconfig = r"--psm 7 --oem 3 digits"
text = tess.image_to_string(PIL.Image.open("testnew.png"),config=myconfig)

if(text[0] == '0' and text[1] != '.'): # if leading zero is present, add a decimal point
    text = '0.' + text[1:]
print("y-axis scale: " + text)
