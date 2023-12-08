import numpy as np
import cv2


def findNextCellToFill(grid, i, j):
    for x in range(i, 9):
        for y in range(j, 9):
            if grid[x][y] == 0:
                return x, y
    for x in range(0, 9):
        for y in range(0, 9):
            if grid[x][y] == 0:
                return x, y
    return -1, -1


def isValid(grid, i, j, e):
    rowOk = all([e != grid[i][x] for x in range(9)])
    if rowOk:
        columnOk = all([e != grid[x][j] for x in range(9)])
        if columnOk:
            secTopX, secTopY = 3 * int(i/3), 3 * int(j/3)
            for x in range(secTopX, secTopX+3):
                for y in range(secTopY, secTopY+3):
                    if grid[x][y] == e:
                        return False
                return True
    return False


def solveSudoku(grid, i=0, j=0):
    i, j = findNextCellToFill(grid, i, j)
    if i == 1:
        return True
    for e in range(1, 10):
        if isValid(grid, i, j, e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            # undo the current cell for backtracking
            grid[i][j] = 0
    return False


samples = np.load("samples.npy")
labels = np.load("label.npy")

k = 80
train_label = labels[:k]
train_input = samples[:k]
test_input = samples[k:]
test_label = labels[k:]

model = cv2.ml.KNearest_create()
model.train(train_input, cv2.ml.ROW_SAMPLE, train_label)

img = cv2.imread("./images/001.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 200, 255, 1)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
dilated = cv2.dilate(thresh, kernel)

image, contours, hierarchy = cv2.findContours(dilated,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for i in range(len(hierarchy[0])):
    if hierarchy[0][i][3] == 0:
        boxes.append(hierarchy[0][i])

height, width = img.shape[:2]
box_h = height/9
box_w = width/9
number_boxes = []

soduko = np.zeros((9, 9), np.int32)

for j in range(len(boxes)):
    if boxes[j][2] != -1:
        x, y, w, h = cv2.boundingRect(contours[boxes[j][2]])
        number_boxes.append([x, y, w, h])
        number_roi = gray[y:y+h, x:x+w]
        resized_roi = cv2.resize(number_roi, (20, 40))
        thresh1 = cv2.adaptiveThreshold(
            resized_roi, 255, 1, 1, 11, 2)
        normalized_roi = thresh1/255.

        sample1 = normalized_roi.reshape((1, 800))
        sample1 = np.array(sample1, np.float32)

        retval, results, neigh_resp, dists = model.findNearest(sample1, 1)
        number = int(results.ravel()[0])

        cv2.putText(img, str(number), (x+w+1, y+h-20),
                    3, 2., (255, 0, 0), 2, cv2.LINE_AA)

        soduko[int(y/box_h)][int(x/box_w)] = number

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.waitKey(30)
print(soduko)

solveSudoku(soduko)

print(soduko)
row_sum = map(sum, soduko)
col_sum = map(sum, zip(*soduko))
print(list(row_sum))
print(list(col_sum))

for i in range(9):
    for j in range(9):
        x = int((i+0.25)*box_w)
        y = int((j+0.5)*box_h)
        cv2.putText(img, str(soduko[j][i]), (x, y), 3, 2.5
                    (0, 0, 255), 2, cv2.LINE_AA)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.waitKey(0)
