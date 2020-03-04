from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy
from numpy import array

#number of times pregnant
#plasma glucose concentration
#diastole bp
#triceps skin fold thickness
#2-hours serum insulin
#BMI
#diabetes pedigree function
#age
#class variable


numpy.random.seed(2)


dataset = numpy.loadtxt("prima-indians-diabetes.csv", delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# create model, add dense layers one by one specifying activation function
model = Sequential()
model.add(Dense(15, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 1000, batch_size=20, validation_data=(x_test, y_test))

# save the model
#model.save('weights.h5')

Xnew = array([[7,150,70,36,0,35,0.6,51]])

ynew = model.predict_classes(Xnew)

print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))