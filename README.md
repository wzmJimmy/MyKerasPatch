# MyKerasPatch
A patch for resumable training with callbacks in Keras.<br>

One may want to train a time-consuming CNN model on a public platform with time limmit on single task. People have to break the total training epochs to several turns, while model saving and loading is needed between different turns. It's really annoying if the number of turns is large. <br>

What makes it worse, Keras's model.save seems mainly designed for transfer learning, where callback status is not saved 
inside. That makes model training with basic callbacks like 'earlyStop','ModelCheckPoint', and 'ReduceLROnPlateau' working 
totally different if we save and load the same model. <br>

In this small patch, I use pickle to save the status of callbacks and build a simple wrapper which makes those status able to 
saved in one class. it also offers 'train' and 'train_generator' functions with parameters similar to model.fit and model.fit_generator, by which the training can be done automatically with @epochs, @start_epoch, and @ep_turn initialized. <br>

With some more user settings, the model can be trainned within the same python file as if it never stop and resume. A sample of using this patch in training NN with TGS salt identification dataset on Euler computing platform is listed in https://github.com/wzmJimmy/NeuralNetwork_Coursework <br>


There is still one problem left: this Patch does not guarantee the model is reproducible, which mainly relys on the random state but we cannot get it from keras easily. <br>
