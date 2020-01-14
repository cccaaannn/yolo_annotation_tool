def __convert_annotations_opencv_to_yolo(image_width, image_height, all_points):
    """
    0 x1 y1 x2 y2 -> 0 centerX centerY width height
    converts opencv image coordinates to yolo style annotations label of the images has to be included
    """
    yolo_points = []
    for points in all_points:
        label = points[0]
        x1 = points[1][0]
        y1 = points[1][1]
        x2 = points[2][0]
        y2 = points[2][1]

        w = abs(x1-x2)
        h = abs(y1-y2)
        
        if(x1>x2):
            cx = x1-w/2
        else:
            cx = x2-w/2
        
        if(y1>y2):
            cy = y1 - h/2
        else:
            cy = y2 - h/2
            
        cx = cx/image_width
        cy = cy/image_height
        
        w = abs(x1-x2)/image_width
        h = abs(y1-y2)/image_height

        yolo_points.append((label, cx,cy,w,h))

    return yolo_points


def __convert_annotations_yolo_to_opencv(image_width, image_height, all_points):
    """
    0 centerX centerY width height -> 0 x1 y1 x2 y2
    converts yolo style annotations to opencv image coordinates so you can draw rectangles
    it also returns label of the image as first element of the list
    """

    # most of the time this values will be reed from a file and an empty list can cause some problems so I used try catch 
    try:
        opencv_points = []
        for points in all_points:
            # percent to abs pixel positions
            mx = image_width * float(points[1])
            my = image_height * float(points[2])

            w = image_width * float(points[3])
            h = image_height * float(points[4])

            # it should be like this in a regular coodinate system but on computer y coordinates are reversed 
            # 0,0 is top left instead of bottom left
            # soly = int(my + h/2) sagy = int(my - h/2) 

            top_left_y = int(my - h/2)
            top_left_x = int(mx - w/2)
            bottom_rigth_y = int(my + h/2)
            bottom_right_x = int(mx + w/2)
            opencv_points.append((int(points[0]), top_left_x, top_left_y, bottom_right_x, bottom_rigth_y))
    except(IndexError):
        pass

    return opencv_points

