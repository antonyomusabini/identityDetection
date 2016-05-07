Identify And Speak
========
Identify And Speak detectef faces and identitys of the people in fromnt on a wabcam.
Then the system chooses one of the mp3s to read it.

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
