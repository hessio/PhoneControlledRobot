##Import necessary libraries
from flask import Flask, render_template, Response, request
import cv2
import json
import pyowm
from datetime import datetime, timedelta
import numpy as np
import cv2
import util as ut

# Enable camera
cap0 = cv2.VideoCapture('/dev/video2')
cap1 = cv2.VideoCapture('/dev/video0')

cap0.set(3, 640)
cap0.set(4, 420)

cap1.set(3, 640)
cap1.set(4, 420)

back = False

ut.init_gpio()
speed = 0

app = Flask(__name__)

@app.route("/index", methods=["GET", "POST"])
def button():
    global back
    if request.data != b'':
        speed = request.data
        ut.set_speed(int(speed))
        print(int(speed))

    if request.method == "POST":
        speed=int(request.form["speed"])
        print(speed)
        if request.form.get('direction') == 'forward':
            ut.set_speed(speed)
            ut.forward()
            back = False
            print("forward")
            return render_template('index.html', speed=speed)
        elif request.form.get('direction') == 'stop':
            speed=20
            ut.set_speed(speed)
            speed=0
            ut.set_speed(0)
            ut.stop()
            print("stop")
            return render_template('index.html', speed=speed)
        elif request.form.get('direction') == 'back':
            ut.set_speed(speed)
            ut.back()
            back = True
            print("back")
            return render_template('index.html', speed=speed)
        elif request.form.get('direction') == 'left':
            ut.set_speed(speed)
            ut.left()
            print("left")
            return render_template('index.html', speed=speed)
        elif request.form.get('direction') == 'right':
            ut.set_speed(speed)
            ut.right()
            print("right")
            return render_template('index.html', speed=speed)

    elif request.method == 'GET':
        return render_template('index.html', speed=20)

def gen_frames():
    while True:
        success, frame = cap0.read()  # read the camera frame
        if not success:
            break
        else:
            while(True):
                if not back:
                    success, img = cap0.read()
                else:
                    success, img = cap1.read()
                imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                #if back:
                #    cap = cap1 #cv2.VideoCapture('/dev/video1')
                ret, buffer = cv2.imencode('.jpg', imgGray)
                frame = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)


'''ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

while(True):
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        print("fart")
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
     ret, buffer = cv2.imencode('.jpg', frame)
     frame = buffer.tobytes()
     yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
