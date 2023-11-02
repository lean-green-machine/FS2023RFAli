import cv2
import numpy as np

image = cv2.imread("ldperotate.png") #Reads in image file
cv2.imshow("Image",image) #Displays image
cv2.waitKey(0) #Awaits user to press key
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #Converts image to grayscale
cv2.imshow("Image",gray) #Displays grayscaled image
cv2.waitKey(0)
indices = np.where(gray != 255) #Returns 2 arrays of y and x coordinates, respectively, of each pixel that is NOT white (255)
#sort indices by x...:
coordinates = list((zip(indices[1],indices[0]))) #zips up x array and y array to make coordinate list
#print(indices)
print(coordinates) #prints list of coordinates