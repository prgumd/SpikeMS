import glob, os, cv2
import numpy as np

# os.mkdir("invertedImgs")
# folder = "invertedImgs"
# os.mkdir(folder)
for file in glob.glob("*.jpg"):
    img = cv2.imread(file, 0)

    invert = cv2.bitwise_not(img)
    resized = cv2.resize(invert, (256,256), interpolation = cv2.INTER_NEAREST)
    # kernel = np.ones((3,3), np.uint8) 
    # erode = cv2.dilate(resized, kernel, iterations=1) 

    # cv2.imwrite(os.path.join("invertedImgs","invert_" + file), resized)
    # cv2.imwrite(os.path.join(folder,"invert_" + file), resized)
    cv2.imwrite("invert_" + file, resized)