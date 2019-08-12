# CounterfeitCurrencyDetection
A web application which detects a counterfeit note when an image of a 500 or 2000 rupee note is given.

The model was trained on a self-generated dataset consisting of 40000 images in total. The dataset consisted of 10000-500 rupee real notes, 10000-500 rupee fake notes, 10000-2000 rupee real notes, 10000-2000 rupee fake notes. For the real notes, different images of notes depicting real-life scenario(like colored notes, something written on notes) were captured. According to RBI, alongwith the Gandhiji watermark, they are 12 more security features which needs to be checked while detecting a counterfeit note. For fake notes, the notes were photoshopped and some of the security features were disturbed in the images. And then the notes were replicated using data augmentation techniques.

The file GlimpseOfDataset shows how the dataset looked like.
The file CNNArchitecture shows how the final architecture of our model looked like.
The file train.py is used for training.
The file test.py was used for final testing.


