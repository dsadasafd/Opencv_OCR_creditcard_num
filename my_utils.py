import cv2

def sort_contours(contours, method="left_to_right"):
    reverse = False
    i = 0
    boundingBoxes = []

    if method == "right_to_left" or method == "bottom_to_top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    for c in contours:
        boundingBoxes.append(cv2.boundingRect(c))
        (new_contours, new_boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda x:x[1][i], reverse=reverse))

    return (new_contours, new_boundingBoxes)

def resize(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    h, w = image.shape[:2]
    dim = ()
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)
    else:
        r = width / float(w)
        dim = (width, int(h*r))
        pass
    new_size_img = cv2.resize(image, dim, interpolation=interpolation)
    return new_size_img




