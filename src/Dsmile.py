import cv2 as cv

def SmileNow():
    # Getting the video
    VideoFeed = cv.VideoCapture(0)
    
    # Loading the haar cascades
    frontline_Harcase = cv.CascadeClassifier("../XmlFile/haarcascade_frontalface_default.xml")
    Smile_Harcase = cv.CascadeClassifier("../XmlFile/haarcascade_smile.xml")
    
    # Processing of the frame
    while True:
        _isReady, frame = VideoFeed.read()
        if not _isReady:
            break
            
        # Converting to gray for fast processing
        Gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Detecting the face first
        detections = frontline_Harcase.detectMultiScale(
            Gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)  # Changed from maxSize to minSize
        )
        
        # Processing each detected face
        for (x, y, w, h) in detections:
            # Drawing rectangle around the face
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 250, 255), 2)  # Fixed rectangle coordinates
            
            # Capture section of the image using image slicing
            Gray_slice = Gray[y:y+h, x:x+w]
            
            # Detect smiles in the face region
            smile_detected = Smile_Harcase.detectMultiScale(
                Gray_slice,
                scaleFactor=1.7,
                minNeighbors=20,
                minSize=(25, 25)
            )
            
            # Drawing rectangles around detected smiles
            for (sx, sy, sw, sh) in smile_detected:
                # The smile coordinates need to be offset by the face position
                cv.rectangle(
                    frame,
                    (x + sx, y + sy),           # Offset by face position
                    (x + sx + sw, y + sy + sh),  # Offset by face position
                    (0, 250, 200),
                    2
                )
        
        # Showing the image
        cv.imshow("smile_finder", frame)
        
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    
    VideoFeed.release()
    cv.destroyAllWindows()

SmileNow()