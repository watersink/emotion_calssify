# emotion_calssify
==========

Description:
----------

an emotion classify demo based on keras Introduction: This is an emotion classify demo ,which based keras(tensorflow)


Test:
----------
"""video test""" 

#first prama:ft1.mp4 is a video ,mp4,avi,any is ok 

#second prama:True will show the video result in axis using curve

    ea = Emotion_analysis()

    ea.video_analysis("ft1.mp4", True)

![image]( https://github.com/watersink/emotion_calssify/raw/master/demo_images/Figure_1.png)


"""image test""" 

    ea = Emotion_analysis() 

    base_dir="./faces/" 

    for img_dir in os.listdir("./faces/"): 

        ea.image_analysis(os.path.join(base_dir,img_dir))

![image]( https://github.com/watersink/emotion_calssify/raw/master/demo_images/demo.jpg)

you can commit one of up,according to you apply,and run this: 

    python3 emotion_calssify.py

