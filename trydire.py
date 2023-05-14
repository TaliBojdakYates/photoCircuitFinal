import cv2
import numpy as np
from ultralytics import YOLO
import time
import random
from defs import *
import math
from numbersDetect.detectNumber import number_detect
# Load a model
model = YOLO()
model = YOLO("train4/weights/best.pt")


image = "91514.jpg"

values = ['V','2', '1', '0', '3', 'I', 'A', '4', '6', '8', '7', '5', '9', 'k', 'M', '.', 'x', 'u', 'm','n']


results = model.predict(source=image)  # can also put ,save=True
results = results[0].boxes
bxes = results.xyxy.tolist() # this holds the bounding box coordinates
clsses = results.cls.tolist() # this holds the classes for the boxes
components = []  # list to hold component information

# boxes, removed_indices = remove_boxes_inside(bxes, 0.85)
# removed_indices.sort(reverse=True)



# # remove elements from the list based on indices
# for i in removed_indices:
#     del clsses[i]

def centerPoint(bx):
    return ((bx[2]-bx[0])/2 + bx[0], (bx[3]-bx[1])/2 + bx[1])

class Component:
    def __init__(self, id, box, classType, value=None, unit=None, center=None, nodes=None):
        self.id = id
        self.box = box
        self.classType = classType
        self.value = value
        self.unit = unit
        self.center = center
        self.nodes = nodes


for c in range(len(bxes)):
    center = centerPoint(bxes[c])
    component = Component(id=c,classType=clsses[c],box=bxes[c],center=center)
    components.append(component)  
      

def distance(point1, point2):
    return math.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)
  
def assignUnit(componentClass, number):
    
    unit = ''
    
    if (componentClass == 0):
        # This is a voltage source
        if not number[-1].isalpha():
            unit = 'V'
        else:
            prefix = number[-2] if len(number) >= 2 and number[-2].isalpha() else number[-1]
            prefix = prefix.lower()
            if prefix == 'u':
                unit = 'u_V'
            elif prefix == 'm':
                unit = 'm_V'
            elif prefix == 'v':
                unit = 'V'
            elif prefix == 'k':
                unit = 'k_V'
            elif prefix == 'm':
                unit = 'M_V'
                
    elif(componentClass == 1):
        if not number[-1].isalpha():
            unit = 'ohm'
        else:
            prefix = number[-2] if len(number) >= 2 and number[-2].isalpha() else number[-1]
            prefix = prefix.lower()
            if prefix == 'u':
                unit = 'uohm'
            elif prefix == 'm':
                unit = 'mohm'
            elif prefix == 'k':
                unit = 'kohm'
            elif prefix == 'm':
                unit = 'Mohm'
    elif(componentClass == 2):
        #current source
        pass

    if(unit == ''):
        prefix = number[-2:].lower() if len(number) >= 2 and number[-2:].isalpha() else number[-1].lower()
        prefix = prefix.lower()
        if prefix == 'uv':
                unit = 'u_V'
        elif prefix == 'mv':
            unit = 'm_V'
        elif prefix == 'v':
            unit = 'V'
        elif prefix == 'kv':
            unit = 'k_V'
        elif prefix == 'mv':
            unit = 'M_V'
    
    digit = ''
    for x in number:
        if x.isdigit():
            digit += x
        else:
            break
   
    return digit, unit
    




numbers = number_detect(image,values)


print(numbers)



# iterate through numbers of components
for componentIndex in range(len(components)):
    closest_value = None
    closest_unit = None
    closest_distance = math.inf
    
    # iterate through numbers to find closest center
    for number in numbers:
        distanceValue = distance(number[1], components[componentIndex].center)
        if distanceValue < closest_distance:
            closest_distance = distanceValue
            closest_value,closest_unit = assignUnit(clsses[componentIndex], number[0])
            

    components[componentIndex].unit = closest_unit
    components[componentIndex].value = closest_value
    

    
img = cv2.imread(image)
imgOrg = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7,7), 1)
# cv2.imshow("cor",img)
(thresh, blackAndWhiteImage) = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)# was lower 245
# cv2.imshow("blk",blackAndWhiteImage)
img = cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_GRAY2RGB)



classes = []
boxes = []
plusArray = []
minusArray = []
arrowArray = []

for com in components:
   
    
    if com.classType == VOLTAGE or com.classType == CURRENT or com.classType == RESISTOR or com.classType == DEPVOLTAGE:
        topCorner = [int(com.box[0])-2,int(com.box[1])-2]
        bottomCorner = [int(com.box[2]+2),int(com.box[3])+2]

        com.box[0] = int(com.box[0]) - 2.0
        com.box[1] = int(com.box[1]) - 2.0
        com.box[2] = int(com.box[2]) + 2.0
        com.box[3] = int(com.box[3]) + 2.0
        boxes.append(com.box)
        classes.append(com.classType)
        
        cv2.rectangle(img, topCorner, bottomCorner, color=(255, 255, 255), thickness=-1)
        cv2.rectangle(imgOrg, topCorner, bottomCorner, color=(0, 0, 255), thickness=1)

    elif com.classType == PLUS:
        topCorner = [int(com.box[0]), int(com.box[1])]
        bottomCorner = [int(com.box[2]), int(com.box[3])]
        plusArray.append(centerPoint(com.box))
        cv2.rectangle(imgOrg, topCorner, bottomCorner, color=(0, 0, 255), thickness=1)

    elif com.classType == MINUS:
        topCorner = [int(com.box[0]), int(com.box[1])]
        bottomCorner = [int(com.box[2]), int(com.box[3])]
        minusArray.append(centerPoint(com.box))
        cv2.rectangle(imgOrg, topCorner, bottomCorner, color=(0, 0, 255), thickness=1)
    elif com.classType == ARROW:
        topCorner = [int(com.box[0]), int(com.box[1])]
        bottomCorner = [int(com.box[2]), int(com.box[3])]
        arrowArray.append(centerPoint(com.box))
        cv2.rectangle(imgOrg, topCorner, bottomCorner, color=(0, 0, 255), thickness=1)

print('p',plusArray)
print(minusArray)
print(arrowArray)
cv2.imshow("yolo",imgOrg)
cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow("hsv",img)
# Threshold of blk in HSV space
lower_blk = np.array([0,0,0])
upper_blk = np.array([360,255,80])
# upper_blk = np.array([170,260,260])

# mask, blur, canny
img = cv2.inRange(img, lower_blk, upper_blk)
# cv2.imshow("mask", img)
img = cv2.GaussianBlur(img, (7,7), 1)
# cv2.imshow("blur", img)
# img = cv2.Canny(img, 200, 200)
img = cv2.Canny(img, 20, 250)
cnts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
counter = 0
print(len(cnts))

countours = []
for cnt in cnts:
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    # print(counter, area, int(peri))
    if area > 50 or peri > 500:
        color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
        cv2.putText(imgOrg, f'n:{counter}', (int(cnt[0][0][0]), int(cnt[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, .7, color)
        # print(counter, colors[counter])
        cv2.fillPoly(imgOrg, pts=[cnt], color=color)
        # peri = cv2.arcLength(cnt,True)
        # approx = cv2.approxPolyDP(cnt, .02*peri, True)
        # objCor = len(approx)
        # x , y , w, h = cv2.boundingRect(approx)
        counter += 1
        countours.append(cnt.tolist())
    # else:
    #     cv2.fillPoly(imgOrg, pts=[cnt], color=(128, 0, 128))


    # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # cv2.putText(imgOrg, f'n:{counter}', (int(cnt[0][0][0]), int(cnt[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, .7,
    #             color)
    # cv2.fillPoly(imgOrg, pts=[cnt], color=color)
    # counter += 1
    # countours.append(cnt.tolist())



# cv2.imshow("canny", img)
cv2.imshow("out1", imgOrg)
cv2.waitKey(0)

# conectionDict = {}
# pointDict = {}
# lowestNode = None
# # loop through the features
# for component in components:
#     coor = boxes[component]
#     # print(coor)
#     if classes[component] == VOLTAGE:
#         print("voltage", component)
#         cv2.putText(imgOrg, f"f{component}", (int(coor[0]), int(coor[3])), cv2.FONT_HERSHEY_SIMPLEX, .7, 255)
#     elif classes[component] == RESISTOR:
#         print("res", component)
#         cv2.putText(imgOrg, f"f{component}", (int(coor[0]), int(coor[3])), cv2.FONT_HERSHEY_SIMPLEX, .7, 255)
#     elif classes[component] == CURRENT:
#         print("curr", component)
#         cv2.putText(imgOrg, f"f{component}", (int(coor[0]), int(coor[3])), cv2.FONT_HERSHEY_SIMPLEX, .7, 255)

#     plusPoint = None
#     minusPoint = None
#     # then loop through the nodes
#     for node in range(len(countours)):
#         cnt = countours[node]
#         # print(cnt)
#         # point in this format [[x y]]
#         for point in cnt:
#             # if a point in the node is within the x and within the y
#             if int(coor[0])-3 < point[0][0] and point[0][0] < int(coor[2])+3 and int(coor[1])-3 < point[0][1] and point[0][1] < int(coor[3])+3:
#                 # print('feature', component, 'node', node) # now have to figure out which distance to use
#                 # print(point[0][0] - (int(coor[0])-5), point[0][1] - (int(coor[1])-5), (int(coor[2])+5) -point[0][0], (int(coor[3])+5)-point[0][1])

#                 # adding to dictionary
#                 if component in conectionDict.keys() and node not in conectionDict[component]:
#                     conectionDict[component].append(node)
#                     pointDict[component].append(point[0])
#                     minusPoint = point[0]
#                 elif component not in conectionDict.keys():
#                     conectionDict[component] = [node,]
#                     pointDict[component] = [point[0], ]
#                     plusPoint = point[0]

#                 # keeping track of lowest node used
#                 if lowestNode == None:
#                     lowestNode = node
#                 elif node < lowestNode:
#                     lowestNode = node

#     # to do the sign dection
#     # loop through the sign list
#     # if signs center point is inside the box of the component
#     #   if distance from center point is closer to point[0] for this node
#             # then change the position of that node accoordingly in the connection dict
#     if classes[component] == CURRENT and len(arrowArray) != 0: # need to make or dependent current
#         for arrow in arrowArray:
#             if int(coor[0])-3 < arrow[0] and arrow[0] < int(coor[2])+3 and int(coor[1])-3 < arrow[1] and arrow[1] < int(coor[3])+3:
#                 if distance(arrow, minusPoint) > distance(arrow, plusPoint):
#                     conectionDict[component] = conectionDict[component][::-1]
#                     pointDict[component] = pointDict[component][::-1]
#                 print('switch arrows')
#                 break
#     else:
#         for plus in plusArray:
#             if int(coor[0])-3 < plus[0] and plus[0] < int(coor[2])+3 and int(coor[1])-3 < plus[1] and plus[1] < int(coor[3])+3:
#                 print('switch arrows')
#                 print(minusPoint)
#                 if distance(plus, minusPoint) < distance(plus, plusPoint):
#                     conectionDict[component] = conectionDict[component][::-1]
#                     pointDict[component] = pointDict[component][::-1]
#                 break
#         else:
#             if len(minusArray) != 0:
#             # look in minus array if we do not detect
#                 for minus in minusArray:
#                     if int(coor[0])-3 < minus[0] and minus[0] < int(coor[2])+3 and int(coor[1])-3 < minus[1] and minus[1] < int(coor[3])+3:
#                         print('switch arrows')
#                         if distance(minus, minusPoint) > distance(minus, plusPoint):
#                             conectionDict[component] = conectionDict[component][::-1]
#                             pointDict[component] = pointDict[component][::-1]
#                         break

# print(conectionDict)
# print(lowestNode)

# for node in pointDict:
#     # print(pointDict[node])
#     cv2.putText(imgOrg, '+', (pointDict[node][0][0]+15, pointDict[node][0][1]+15),cv2.FONT_HERSHEY_SIMPLEX, .7, 255)
#     cv2.putText(imgOrg, '-', (pointDict[node][1][0]+15, pointDict[node][1][1]+15),cv2.FONT_HERSHEY_SIMPLEX, .7, 255)




# cv2.imshow("out1", imgOrg)
# cv2.waitKey(0)


# componentValues = []
# for inner_list in components:
#     digits = ""
#     for char in inner_list[1]:
#         if char.isdigit():
#             digits += char
#     componentValues.append([digits, inner_list[2]])

# print(componentValues)

# import PySpice
# import os
# import PySpice.Logging.Logging as Logging
# logger = Logging.setup_logging()
# from PySpice.Spice.Netlist import Circuit
# from PySpice.Unit import *

# # from PySpice.Spice.Simulation import Transient
# # from PySpice.Spice.Simulation import dc
# # conectionDict = {0: [4, 5], 1: [1, 3], 2: [2, 4], 3: [2, 3], 4: [1, 5], 5: [1, 2], 6: [3, 4], 7:[5,1]}
# # classes = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
# # lowestNode = 1
# circuit = Circuit('Resistor Bridge')

# # replace the biggest node number with 0
# for node in conectionDict:

#     if lowestNode in conectionDict[node]:
#         conectionDict[node][conectionDict[node].index(lowestNode)] = 0
# print(conectionDict)
# print(classes)
# for i in range(len(classes)):

#     if classes[i] == 0.0:
#         print('voltage source \n')
#         val = 10
#         circuit.V(i,conectionDict[i][0],conectionDict[i][1],val)
#     elif classes[i] == 1.0:
#         print('resistor\n')
#         val = 5
#         circuit.R(i,conectionDict[i][0],conectionDict[i][1],val)
#     elif classes[i] == 2.0:
#         print('dependent voltage source\n')
#     elif classes[i] == 3.0:
#         print('current source\n')
#         val = 2
#         circuit.I(i,conectionDict[i][0],conectionDict[i][1],val)
#     elif classes[i] == 4.0:
#         print('dependent current source\n')
# # print(circuit)

# els = [i for i in circuit.element_names]
# for i in els:
#     if 'R' in i:
#         # print(i)
#         # r = eval(f'circuit.{i}.plus.add_current_probe(circuit)')
#         # circuit.r.plus.add_current_probe(circuit)
#         circuit[i].plus.add_current_probe(circuit)


# simulator = circuit.simulator(temperature=25, nominal_temperature=25)
# analysis = simulator.operating_point()

# for node in analysis.branches.values():
#     print('Node {}: {:5.2f} A'.format(str(node), float(node)))

# for node in analysis.nodes.values():
#     print('Node {}: {:4.1f} V'.format(str(node), float(node)))
# cv2.imshow("out1", imgOrg)
# cv2.waitKey(0)

# # todo need a way to determine the closest line to a feature
# # todo gets confused when more symbols on the circuit, when too much noise

# # todo for user
# # make sure some white space between circuit and edge
# # best if done on plain light background
# # remove extra symbols, just values of the components
