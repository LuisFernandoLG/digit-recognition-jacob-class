import cv2
import numpy as np
from tensorflow import keras

new_model = keras.models.load_model('model.h5')

# a function that capture video and retuns frame
def capture():
    # import the opencv library
    import cv2
    # define a video capture object1
    vid = cv2.VideoCapture(0)
    while(True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        
        if(frame is not None):
            # Display the resulting frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
            ret3,th3 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            invert = 255 - th3
            
            # get contours of image
            contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(frame, contours, -1, (0,255,0), 5)
            
            #aqui se dibujan los contornos
            
            #draw a box around the number
            for i in range(len(contours)):
                x,y,w,h = cv2.boundingRect(contours[i])
                area = cv2.contourArea(contours[i])
                #no includ small areas
                if area > 300:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
                    # add a little padding to the image

                    # crop the image
                    crop = invert[y:y+h, x:x+w]
                    
                    # add border to the image
                    crop = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0,0,0])

                    filename = f"holi.png"
                    cv2.imwrite(filename, crop)
                    
                    
                                    
                    roi_with_padding = cv2.resize(crop, (28, 28))
                    #predict number by roi image
                    roi_with_padding = roi_with_padding.reshape(1, 28, 28, 1)
                    roi_with_padding = roi_with_padding.astype('float32')
                    roi_with_padding /= 255
                    
                    #array
                    pred = new_model.predict(roi_with_padding)
                    index = np.argmax(pred[0])
                    s1 = str(index)
                    # s2 = str( pred[0][index])
                    s2 = ""

                    # cv2.drawContours(frame, contours, -1, (255,255,255), 2)
                    cv2.putText(frame, s1+s2 , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('frame', frame)
            
                
                

                

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    vid.release()
    cv2.destroyAllWindows()

capture()