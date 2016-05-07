Identify And Speak
========
Identify And Speak detectes faces and identities of the people in front on a wabcam.

Then, according to the sitations such as the number of faces in front of the camera, the identifies of them, the system chooses one of the mp3s to run.

#### 0. Dependencies :

OpenCV

#### 1. Install

```
> make
```

#### 2. Run

```
> ./identifyAndSpeak --cascade="/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml" --nested-cascade="/usr/share/opencv/haarcascades/haarcascade_eye.xml" --scale=1.3
```
