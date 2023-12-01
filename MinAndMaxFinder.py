import cv2
import numpy as np
from scipy.signal import find_peaks

# Function to find the "average" point for every x coordinate of a coordinate
# tuple list and return a tuple list without the duplicate points
def average_points(points):
    # Step 1: Create a dictionary where the keys are the x-coordinates and the values are lists of y-coordinates
    x_dict = {}
    for x, y in points:
        if x not in x_dict:
            x_dict[x] = []
        x_dict[x].append(y)

    # Step 2: For each key in the dictionary, if there is more than one y-coordinate, calculate the average y-coordinate
    for x in x_dict:
        if len(x_dict[x]) > 1:
            x_dict[x] = sum(x_dict[x]) / len(x_dict[x])

    # Step 3: Convert the dictionary back into a list of tuples
    return [(x, y) for x, y in x_dict.items()]

np.set_printoptions(threshold=9999)

image = cv2.imread("PVC.png") # Reads in image file
cv2.imshow("Input Image",image) # Displays image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # Converts image to grayscale
gray = cv2.flip(gray,0)
indices = np.where(gray < 235) # Returns 2 arrays of y and x coordinates, respectively, of each pixel that is NOT white (255)

points = list((zip(indices[1],indices[0]))) # Zips up x array and y array to make coordinate list
# Sort the list of tuples in place by the first element of each tuple
points.sort(key=lambda x: x[0])

avgpoints = average_points(points)

# Iterate through tuple list and convert
i = 0
while(i<len(avgpoints)):
  if isinstance(avgpoints[i][1],list):
    x = list(avgpoints[i])
    x[1] = x[1][0]
    avgpoints[i] = tuple(x)
  i+=1

#Convert the list of tuples into two separate lists for the x and y coordinates
x_coords = [point[0] for point in avgpoints]
y_coords = [point[1] for point in avgpoints]

# Convert the list of y-coordinates into a numpy array
y_coords_array = np.array(y_coords)

# Use the argrelextrema function to find the indices of the local maxima and minima
max_indices,_ = find_peaks(y_coords_array, prominence=2.5)
min_indices,_ = find_peaks(-y_coords_array, prominence=2.5)

# Extract the local maxima and minima from the list of points using the indices
local_maxima = [avgpoints[i] for i in max_indices]
local_minima = [avgpoints[i] for i in min_indices]

# Print the local maxima and minima
print("# Maxima found: " + str(len(local_maxima)))
print("Local maxima:", local_maxima)
print("Global maximum:", max(local_maxima, key=lambda x: x[1]))
print("# Minima found: " + str(len(local_minima)))
print("Local minima:", local_minima)
print("Global minimum:", min(local_minima, key=lambda x: x[1]))

# Our y-values are sometimes of type float. For purposes of drawing circles at the points, firstly must convert to int
local_maxima = [(i, int(j)) for i, j in local_maxima]
local_minima = [(i, int(j)) for i, j in local_minima]

image = cv2.flip(image,0)

for coord in local_maxima:
    cv2.circle(image, coord, 3, (0,255,0), -1)
for coord in local_minima:
    cv2.circle(image, coord, 3, (255, 0, 0), -1)
cv2.imshow("Maximas (green) & Minimas (blue)",cv2.flip(image,0))
cv2.waitKey(0)