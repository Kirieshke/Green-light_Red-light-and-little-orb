import cv2
class VideoOrb:

    def find_keypoints(video_path):
    # Open the video file
        video = cv2.VideoCapture(video_path)
        
        # Check if the video file was opened successfully
        if not video.isOpened():
            print("Error opening video file")
            return
        
        # Create an ORB object
        orb = cv2.ORB_create()
        
        while True:
            # Read the next frame from the video
            ret, frame = video.read()
            
            # If the frame was not read successfully, we have reached the end of the video
            if not ret:
                break
            
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find keypoints and compute descriptors
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            # Draw keypoints on the frame
            frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            print(frame_with_keypoints)
            # Display the frame with keypoints
            cv2.imshow("Frame with Keypoints", frame_with_keypoints)
            
            # Wait for the 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the video file and close the output window
        video.release()
        cv2.destroyAllWindows()