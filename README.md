# MachineLearning_Hottnesscale

Hottnesscale uses thousands of faces images as training data. Every face is marked either as beautiful or not beautiful, and the every face has its facial data i.e. landmarks, like the distance between the eyes, or the size of the mouth, there are 52 landmarks every face. Accordingly, I'm using K-Means algorithm to get 10 average faces(which are essentially landmarks data) as perfect face. Then when a new person take a photo and send the image to the python script, a score will be calculated comparing to each of the ten perfect "face", and the highest one will be the final sore.  

[![See Video Here](https://github.com/suhan1996/MachineLearning_Hottnesscale/blob/master/pre.png?raw=true)](https://www.youtube.com/watch?v=UXsIwuZLUTU&t=270s)
