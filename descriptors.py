import cv2 as cv

img1 = cv.imread('saw1.jpg', 0)
sift = cv.SIFT_create()
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

key_points1, descriptors1 = sift.detectAndCompute(img1, None)


def get_descriptors(img2):
    key_points2, descriptors2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=None,
        flags=2
    )

    img3 = cv.drawMatches(img1, key_points1, img2, key_points2, good, None, **draw_params)
    scale_percent = 20
    width = int(img3.shape[1] * scale_percent / 100)
    height = int(img3.shape[0] * scale_percent / 100)
    dim = (width, height)
    img3 = cv.resize(img3, dim, interpolation=cv.INTER_AREA)
    return img3


for i in range(2, 5):
    img2 = cv.imread(f'saw{i}.jpg', 0)
    img3 = get_descriptors(img2)

    cv.imshow(f"saw{i}", img3)
    cv.waitKey(0)
    cv.destroyAllWindows()

video = cv.VideoCapture('sawmovie.mp4')
ret, frame = video.read()

i = 0
while ret:
    i += 1
    if i < 50:
        ret, frame = video.read()
        continue
    key = cv.waitKey(1)
    frame = get_descriptors(frame)
    cv.imshow("sawmovie.mp4", frame)
    if key == ord('q'):
        break
    ret, frame = video.read()

video.release()
cv.destroyAllWindows()