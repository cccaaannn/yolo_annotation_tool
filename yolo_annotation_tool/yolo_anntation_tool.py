import os
import cv2 
import numpy as np
import argparse
import sys


from file_operations import __read_from_file, __write_to_file
from convert_annotations import __convert_annotations_opencv_to_yolo, __convert_annotations_yolo_to_opencv


def yolo_annotation_tool(images_path, class_names_file, max_windows_size=(1200,700), image_extensions = [".jpg", ".JPG", ".jpeg", ".png", ".PNG"]):
    """
    annotation tool for yolo labeling
    
    # warning it uses two global variables (__coords__, __drawing__) due to the opencvs mouse callback function

    # usage 
    a go backward
    d go forward
    s save selected annotations
    z delete last annotation
    r remove unsaved annotations
    c clear all saved annotations
    """
    


    # read images
    images = os.listdir(images_path)
    images.sort()

    # remove not included files
    for image in images:
        image_name, image_extension = os.path.splitext(image)
        if image_extension not in image_extensions: 
            images.remove(image)        

    # add paths to images
    images = [os.path.join(images_path, image) for image in images]

    # read class names
    class_names = __read_from_file(class_names_file)
    class_names = class_names.split()


    # -----unused-----
    def __on__trackbar_change(image):
        """
        Callback function for trackbar
        """
        pass
        
    def __resize_with_aspect_ratio(image, width, height, inter=cv2.INTER_AREA):
        """
        resize image while saving aspect ratio
        """
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if h > w:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        image = cv2.resize(image, dim, interpolation=inter)

        return image
    # ---------------



    def __draw_rectangle_on_mouse_drag(event, x, y, flags, param):
        """
        draws rectangle with mouse events
        """
        global __coords__, __drawing__

        if event == cv2.EVENT_LBUTTONDOWN:
            __coords__ = [(x, y)]
            __drawing__ = True
        
        elif event == 0 and __drawing__:
            __coords__[1:] = [(x, y)]
            im = image.copy()
            cv2.rectangle(im, __coords__[0], __coords__[1], (255, 0, 0), 2)
            cv2.imshow(window_name, im) 

        elif event == cv2.EVENT_LBUTTONUP:
            # __coords__.append((x, y))
            __coords__[1:] = [(x, y)]
            __drawing__ = False
    
            cv2.rectangle(image, __coords__[0], __coords__[1], (255, 0, 0), 2)

            # add points
            points.append(((label),__coords__[0],__coords__[1]))


        elif event == cv2.EVENT_RBUTTONDOWN:
            pass

    def __save_annotations_to_file(image_path, yolo_labels_lists, write_mode):
        """
        saves yolo annnotations lists to annotations file list of lists:[[0,1,1,1,1],[1,0,0,0,0]]
        returns annotation_file_path 
        """

        # prepare the annotations
        yolo_labels = []
        for yolo_labels_list in yolo_labels_lists:
            yolo_labels.append("{0} {1:.6} {2:.6} {3:.6} {4:.6}".format(yolo_labels_list[0], yolo_labels_list[1], yolo_labels_list[2], yolo_labels_list[3], yolo_labels_list[4]))

        image_name, image_extension = os.path.splitext(image_path)
        annotation_file_path = "{0}.txt".format(image_name)

        # if last character of the file is not \n we cant append directly we should add another line 
        # since __write_to_file function writes lists to line inserting an empty string automatically creates a new line
        if(os.path.exists(annotation_file_path)):
            temp_file_content = __read_from_file(annotation_file_path)
            if(temp_file_content):
                if(temp_file_content[-1][-1] != "\n"):
                    yolo_labels.insert(0,"")

        # write prepared annotations to file
        __write_to_file(yolo_labels, annotation_file_path, write_mode=write_mode)

        return annotation_file_path

    def __load_annotations_from_file(image_path):
        """
        loads an images annotations with using that images path returns none if annotation is not exists
        """
        # checking if the annotation file exists if exists read it
        image_name, image_extension = os.path.splitext(image_path)
        annotation_file_path = "{0}.txt".format(image_name)
        if os.path.exists(annotation_file_path):
            annotations = __read_from_file(annotation_file_path)
            annotations = annotations.split("\n")
            annotations = filter(None, annotations)  # delete empty lists 
            annotations = [annotation.split() for annotation in annotations]
            # convert annotations to float and label to int
            # yolo annotation structure: (0 0.8 0.8 0.5 0.5)
            for annotation in annotations:
                annotation[0] = int(annotation[0])
                annotation[1] = float(annotation[1])
                annotation[2] = float(annotation[2])
                annotation[3] = float(annotation[3])
                annotation[4] = float(annotation[4])
            return annotations
        else:
            return None

    def __draw_bounding_boxes_to_image(image_path, class_names):
        """
        draw annotations if file is exists
        """

        # loading annotation file if exists
        annotations = __load_annotations_from_file(image_path)

        if(not annotations):
            return None, 0

        # loading image 
        image = cv2.imread(image_path)
        
        # get dimensions of image
        image_height = np.size(image, 0)
        image_width = np.size(image, 1)

        # convert points
        opencv_points = __convert_annotations_yolo_to_opencv(image_width, image_height, annotations)

        # draw the rectangles using converted points
        for opencv_point in opencv_points:
            # give error if an annoted file has impossible class value
            if(opencv_point[0] > len(class_names)-1):
                raise ValueError("this txt file has an annotation that has bigger class number than current selected class file") 

            cv2.rectangle(image, (opencv_point[1], opencv_point[2]), (opencv_point[3], opencv_point[4]), (0,200,100), 2)
            cv2.line(image, (opencv_point[1], opencv_point[2]), (opencv_point[3], opencv_point[4]), (255, 0, 0), 1) 
            cv2.putText(image, "{0}".format(class_names[opencv_point[0]]), (opencv_point[1], opencv_point[2]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=(0, 0, 0), thickness=2)

        return image, len(annotations)

    def __refresh_image(image_index, label):
        """
        if annotation file exists draw the rectangles resize and return the image if not just return the resized image
        also draw information to the image
        """
        image, annoted_object_count = __draw_bounding_boxes_to_image(images[image_index], class_names)
        if(image is None):
            image = cv2.imread(images[image_index])
        # image = __resize_with_aspect_ratio(image, max_windows_size[0], max_windows_size[1])
        image = cv2.resize(image, max_windows_size)
        
        if(annoted_object_count == 0):
            __save_annotations_to_file(images[image_index], [], "w")

        # show some info with puttext
        cv2.putText(image, "{0}/{1} objects:{2} label: {3}".format(len(images), image_index+1, annoted_object_count, class_names[label]), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0, 200, 100), thickness=2)

        return image 


    

    points = []
    image_index = 0
    label_temp = 0
    global __drawing__
    __drawing__ = False
    window_name = "Yolo annotation tool"


    # create window and set it up
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 40,30)
    cv2.setMouseCallback(window_name, __draw_rectangle_on_mouse_drag,image)
    cv2.createTrackbar('label', window_name, 0, len(class_names)-1, __on__trackbar_change)
    image = __refresh_image(image_index, 0)

    # gui loop
    while(True):

        label = cv2.getTrackbarPos('label', window_name)
        
        # bu ne salak yontem lan kafan mi iyidi yaparken
        if(label != label_temp):
            image = __refresh_image(image_index, label)
            label_temp = label

        # dont refresh the original frame while drawing
        if(not __drawing__):
            cv2.imshow(window_name, image)  
        
        key = cv2.waitKey(30)
        


        # save selected annotations to a file
        if(key == ord("s")):
            if(len(points) > 0):

                image_height = np.size(image, 0)
                image_width = np.size(image, 1)

                # convert and save annotations to file
                yolo_labels_lists = __convert_annotations_opencv_to_yolo(image_width,image_height,points)
                __save_annotations_to_file(images[image_index], yolo_labels_lists, "a")

                # reset points and refresh image
                image = __refresh_image(image_index, label)
                points = []

                print("annotation saved {0}".format(yolo_labels_lists))
        
                


        # move backward
        if(key == ord("a")):
            if(image_index > 0):
                image_index -= 1
                image = __refresh_image(image_index, label)
                points = []

        # move forward
        if(key == ord("d")):
            if(image_index < len(images)-1):
                image_index += 1
                image = __refresh_image(image_index, label)
                points = []

        # delete last annotation
        if(key == ord("z")):
            # load annotations
            yolo_labels_lists = __load_annotations_from_file(images[image_index])            
            if(yolo_labels_lists):
                # delete last one
                yolo_labels_lists.pop()
                # save new annotations (last one deleted)
                annotation_file_path = __save_annotations_to_file(images[image_index], yolo_labels_lists, "w")
                image =__refresh_image(image_index, label)
                points = []

                # # if file is empty delete it
                # if(len(yolo_labels_lists) == 0):
                #     os.remove(annotation_file_path)

        # refresh current image
        if(key == ord("r")):
            image =__refresh_image(image_index, label)
            points = []

        # clear annotations
        if(key == ord("c")):
            __save_annotations_to_file(images[image_index], [], "w")
            image = __refresh_image(image_index, label)        
            points = []


        # if window is closed break this has to be after waitkey
        if (cv2.getWindowProperty(window_name, 0) < 0):
            # cv2.destroyAllWindows()
            break

        # quit on esc
        if(key == 27):
            break


    cv2.destroyAllWindows()



my_parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
my_parser.add_argument('images_path', metavar='images_path', type=str, help='path of your images')
my_parser.add_argument('classes_path', metavar='classes_path', type=str, help='path of your classes file')
my_parser.add_argument('-d', action='store', type=str, help='window dimensions w,h')
my_parser.add_argument('-e', action='store', type=str, help='image extensions')


# Execute the parse_args() method
args = my_parser.parse_args()

image_path = args.images_path
classes_path = args.classes_path
dimensions = args.d
extensions = args.e


max_windows_size=(1200,700)
image_extensions = [".jpg", ".JPG", ".jpeg", ".png", ".PNG"]

if not os.path.isdir(image_path):
    print('images path does not exist')
    sys.exit()

if not os.path.isfile(classes_path):
    print('classes file path does not exist')
    sys.exit()

if(dimensions):
    try:
        dimensions = dimensions.split(",")
        w = int(dimensions[0])
        h = int(dimensions[1])
        dimensions = (w,h)
        max_windows_size = dimensions
    except(ValueError):
        print('window dimensions has to be this format 1100,900')
        sys.exit()

if(extensions):
    try:
        extensions = extensions.split(",")
        image_extensions = extensions
    except(ValueError):
        print('window dimensions has to be this format .jpg,.png')
        sys.exit()


yolo_annotation_tool(image_path, classes_path, max_windows_size=max_windows_size, image_extensions=image_extensions)