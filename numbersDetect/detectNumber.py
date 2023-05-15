from ultralytics import YOLO
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def number_detect(image_path,values):
    model = YOLO()
    model = YOLO("train5/weights/best.pt")

    image = image_path

    results = model.predict(source=image, save=True)  # can also put ,save=True

    results = results[0].boxes
    boxesOriginal = results.xyxy.tolist() # this holds the bounding box coordinates
    classes = results.cls.tolist() # this holds the classes for the boxes

    centers = []
    i = 0

    for box in boxesOriginal:
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        centers.append([center_x,center_y, classes[i]])
        i+=1

    def group_boxes(boxes):
        X = np.array(boxes)
        scores = []
        for n_clusters in range(2, len(boxes)):
            kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, tol=1e-6, n_init=20)
            y_kmeans = kmeans.fit_predict(X)
            score = silhouette_score(X, y_kmeans)
            scores.append(score)
        best_n_clusters = np.argmax(scores) + 2

        kmeans = KMeans(n_clusters=best_n_clusters, max_iter=1000, tol=1e-6, n_init=20)
        y_kmeans = kmeans.fit_predict(X)

        groups = [[] for i in range(best_n_clusters)]

        for i in range(len(y_kmeans)):
            label = y_kmeans[i]
            groups[label].append([boxes[i],boxes[i][2]])

        # sort boxes within each group based on x-coordinate
        for i in range(best_n_clusters):
            groups[i] = sorted(groups[i], key=lambda box: box[0])
 
        return groups
    
    groups = group_boxes(centers)
    
    def map_groups(groups, values):
        groupMap = []
        for group in groups:
            numberValue = ''
            left_x = group[0][0][0]
            left_y = group[0][0][2]
            right_x = group[-1][0][0]
            right_y = group[-1][0][-2]
            center_group_x = (left_x + right_x) / 2
            center_group_y = (left_y + right_y) / 2
            
            for x in group:
                
               
                i = x[1]
                if i == 0:
                    numberValue += values[0]
                elif i == 1:
                    numberValue += values[1]
                elif i == 2:
                    numberValue += values[2]
                elif i == 3:
                    numberValue += values[3]
                elif i == 4:
                    numberValue += values[4]
                elif i == 5:
                    numberValue += values[5]
                elif i == 6:
                    numberValue += values[6]
                elif i == 7:
                    numberValue += values[7]
                elif i == 8:
                    numberValue += values[8]
                elif i == 9:
                    numberValue += values[9]
                elif i == 10:
                    numberValue += values[10]
                elif i == 11:
                    numberValue += values[11]
                elif i == 12:
                    numberValue += values[12]
                elif i == 13:
                    numberValue += values[13]
                elif i == 14:
                    numberValue += values[14]
                elif i == 15:
                    numberValue += values[15]
                elif i == 16:
                    numberValue += values[16]
                elif i == 17:
                    numberValue += values[17]
                elif i == 18:
                    numberValue += values[18]
                elif i == 19:
                    numberValue += values[19]
            groupMap.append([numberValue,[center_group_x,center_group_y]])
        return groupMap
    return map_groups(groups,values)



