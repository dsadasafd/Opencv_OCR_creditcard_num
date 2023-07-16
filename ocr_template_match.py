import  cv2
import numpy as np
import argparse

import my_utils
import my_utils as mu

# 设置整体程序运行的参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--template", required=True, help="path to input template")
args = vars(ap.parse_args())
print(args)

# 绘图展示
def cv_show(name, data):
    cv2.imshow(name, data)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

# 读取模板图像并转换为二值图像，findcontours
img_template = cv2.imread(args['template'])
cv_show("template", img_template)
img_template_cvt = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
cv_show("img_tem_cvt", img_template_cvt)
img_template_cvt_thresh = cv2.threshold(img_template_cvt, 127, 255, cv2.THRESH_BINARY_INV)[1]
cv_show("tem_thresh", img_template_cvt_thresh)

# 对模板轮廓检测
contours, hierarchy = cv2.findContours(img_template_cvt_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 在原template图画出轮廓信息
cv2.drawContours(img_template, contours, -1, (0,0,255), 3)
cv_show("", img_template)
sorted_contours = mu.sort_contours(contours)[0]
#print(sorted_contours)   #测试是否返回按顺序排好的轮廓元组

digits: dict = {}
for i, c in enumerate(sorted_contours):
    x, y, w, h = cv2.boundingRect(c)
    roi = img_template_cvt_thresh[y:y+h, x:x+w]
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi
    #cv_show("",roi)
    pass

#######  对输入图像处理
credit_card = cv2.imread(args['image'])
credit_card = my_utils.resize(credit_card, width=300)
cv_show("", credit_card)
credit_card_cvt = cv2.cvtColor(credit_card, cv2.COLOR_BGR2GRAY)
cv_show("", credit_card_cvt)
credit_resized = my_utils.resize(credit_card_cvt, width=300)
cv_show("", credit_resized)
# tophat and blackhat operation
rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
credit_resized_tophat = cv2.morphologyEx(credit_resized, cv2.MORPH_TOPHAT, rectkernel)
res = np.hstack((credit_resized, credit_resized_tophat))
#cv_show("", res)
gradx = cv2.Sobel(credit_resized_tophat, -1, 1, 0, ksize=3)
cv_show("",gradx)
gradx_abs = np.absolute(gradx)
min_gradx_abs, max_gradx_abs = (np.min(gradx_abs), np.max(gradx_abs))
gradx = (255*((gradx_abs-min_gradx_abs)/(max_gradx_abs-min_gradx_abs)))
gradx = gradx.astype("uint8")
cv_show("", gradx)
# 闭运算，把数字块连在一起，所以先膨胀后腐蚀
gradx = cv2.morphologyEx(gradx, cv2.MORPH_CLOSE, rectkernel)
cv_show("", gradx)
thresh = cv2.threshold(gradx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show("thresh", thresh)
# 继续n个闭运算直到满意图像
n = 1
for i in range(n):
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqkernel)
    pass
cv_show("", thresh)
# 在二值图寻找轮廓位置，再将它画在原图上
thresh_contours, thresh_hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
credit_card_copy = credit_card.copy()
cv2.drawContours(credit_card_copy, thresh_contours, -1, (0,0,255), 3)
cv_show("", credit_card_copy)
# 过滤一些无用的轮廓
cnts = thresh_contours
locs = []
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = float(w)/h
    if 2.5 < ar and ar < 4.5:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x,y,w,h))
    pass
# 对边界矩形排序，left to right
locs = sorted(locs, key=lambda x:x[0])
#print(locs)
# 将目标图像中的数字单独提取出来
output = []
for (i, (gx,gy,gw,gh)) in enumerate(locs):
    groupOutput = []
    group = credit_card_cvt[gy-5:gy+gh+5, gx-5:+gx+gw+5]
    _, group = cv2.threshold(group, 127, 255, cv2.THRESH_BINARY)
    cv_show("", group)
    digit_cnt, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_cnt = my_utils.sort_contours(digit_cnt)[0]
    #print(len(digit_cnt))
    pass
    # 将每个数字抽取出来
    for c in digit_cnt:
        (x,y,w,h) = cv2.boundingRect(c)
        roi = group[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57, 88))
        #cv_show("", roi)
        # 用score记录匹配度得分
        score = []

        for key in digits:
            tem = digits[key]
            result = cv2.matchTemplate(roi, tem, cv2.TM_CCOEFF)
            (_, max_score, _, _) = cv2.minMaxLoc(result)
            score.append(max_score)

        groupOutput.append(str(np.argmax(score)))
        #print(groupOutput)


    cv2.rectangle(credit_card, (gx-5, gy-5), (gx+gw+5, gy+gh+5), (0,0,255), 2)
    cv2.putText(credit_card, "".join(groupOutput), (gx, gy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
    output.extend(groupOutput)
    pass

print("credit card numbers is {}".format("".join(output)))
credit_card_nor = my_utils.resize(credit_card, width=600)
cv2.imshow("", credit_card_nor)
cv2.waitKey(0)
cv2.destroyAllWindows()








