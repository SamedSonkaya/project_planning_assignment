#Samed Sonkaya
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from PIL import Image

var1=0   

def egitim():
    global train_X,train_Y,test_X,test_Y,num_classes
    (train_X,train_Y),(test_X,test_Y)=cifar10.load_data()
    train_x=train_X.astype('float32')
    test_X=test_X.astype('float32')
 
    train_X=train_X/255.0
    test_X=test_X/255.0
    textLabel("cifar10 verileri(train_X,train_Y),(test_X,test_Y)e yuklendi")


    train_Y=np_utils.to_categorical(train_Y)
    test_Y=np_utils.to_categorical(test_Y)
 
    num_classes=test_Y.shape[1]
    
    model=Sequential()


    return (train_X,train_Y),(test_X,test_Y),num_classes,model,train_Y,test_Y

def train():
    global model
    num_classes=test_Y.shape[1]
    model=Sequential()
    model.add(Conv2D(32,(3,3),input_shape=(32,32,3),
                     padding='same',activation='relu',
                     kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_constraint=MaxNorm(3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512,activation='relu',kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))



    sgd=SGD(lr=0.01,momentum=0.9,decay=(0.01/25),nesterov=False)
 
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])



    model.summary()
    
    model.fit(train_X,train_Y,
              validation_data=(test_X,test_Y),
              epochs=10,batch_size=32)


    _,acc=model.evaluate(test_X,test_Y)
    print(acc*100)


    print("Test accuracy:", acc)
    model.save("model1_cifar_10epoch.h5")
    var3=("Test accuracy: %",acc)
    textLabel(var3)
    test1(model)
    

    
    return 

def test1(var1):
    global var2
    if(var1==0):
        root= tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename()
        results={
            0:'aeroplane',
            1:'automobile',
            2:'bird',
            3:'cat',
            4:'deer',
            5:'dog',
            6:'frog',
            7:'horse',
            8:'ship',
            9:'truck'
            }
        im=Image.open(file_path)
        im=im.resize((32,32))
        im=np.expand_dims(im,axis=0)
        im=np.array(im)
        pred=var2.predict_classes([im])[0]
        print(pred,results[pred])
        textLabel(results[pred])
        return
    else:
        var2=var1
        return
def test():
    test1(0)
    return            

def textLabel(var2):
    kod_bolum.config(text=var2)
    return
    

master = Tk()                           
master.title('Deep Learning Project')
canvas = Canvas(master, heigh=450, width=750 )
canvas.pack()

frame_ust = Frame(master, bg='dark gray')
frame_ust.place(relx=0.1,rely=0.1,relwidth=0.8,relheight=0.1)     # ust frame

frame_alt_sol = Frame(master, bg='dark gray')
frame_alt_sol.place(relx=0.1,rely=0.21,relwidth=0.23,relheight=0.7)  #alt sol frame

frame_alt_sag = Frame(master, bg='dark gray')
frame_alt_sag.place(relx=0.34,rely=0.21,relwidth=0.56,relheight=0.7)  #alt sag frame

ana_baslik_etiket = Label(frame_ust, bg='dark gray', text = "Görüntü Sınıflandırma - Derin Öğrenme Projesi", font ="Verdana 10 bold")
ana_baslik_etiket.pack(padx=10, pady=10, side=LEFT)

egitim_button=Button(frame_alt_sol, text="Egitime Verileri Yukle", command=egitim) #egitim veri yukleme butonu
egitim_button.place(relx=0.1,rely=0.2,relwidth=0.8,relheight=0.1)

train_button=Button(frame_alt_sol, text="Egitimi Baslat", command=train) #egitime butonu
train_button.place(relx=0.1,rely=0.4,relwidth=0.8,relheight=0.1)

test_button=Button(frame_alt_sol, text="Test Resmi Yukle", command=test)  #test butonu
test_button.place(relx=0.1,rely=0.6,relwidth=0.8,relheight=0.1)

kod_bolum=Label(frame_alt_sag,bg='white',text="")
kod_bolum.place(relx=0.1,rely=0.1,relwidth=0.8,relheight=0.8)



sag_baslik_etket= Label(frame_alt_sag, bg='dark gray', text = "Kod Paneli", font ="Verdana 10 bold")
sag_baslik_etket.pack(padx=10, pady=10)


master.mainloop()
