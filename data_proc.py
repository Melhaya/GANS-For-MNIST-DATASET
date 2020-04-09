from numpy import *
from numpy.random import rand, randint, randn, random
from matplotlib import pyplot as plt


######################################## LOAD AND PREPARE MNIST IMAGES ########################################

def Nomalize_data(trainX):
	
	# expand to 3d, e.g. add channels dimension
	X = expand_dims(trainX, axis=-1)
	# convert from unsigned ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [0,1]
	X = X / 255.0
	
	return X

######################################## DATA FOR STANDALONE DISCRIMINATOR MODEL ########################################

def generate_train_data(trainX, trainy):
	
	trainX_real = Nomalize_data(trainX)
	trainY_real = ones((len(trainy), 1))

	trainX_fake = rand(28 * 28 * len(trainX_real))
	trainX_fake = trainX_fake.reshape((len(trainX_real), 28, 28, 1))
	trainY_fake = zeros((len(trainy), 1))

	TrainX = concatenate((trainX_real,trainX_fake))
	TrainY = concatenate((trainY_real,trainY_fake))

	return TrainX, TrainY

######################################## VISUALIZE MNIST DATASET ########################################

def Visualize_MNIST(trainX):
	
	# Visualize samples of the dataset
	for i in range(25):
		# define subplot
		plt.subplot(5, 5, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(trainX[i], cmap='gray_r')
	plt.show()


######################################## PLOT ACCURACY AND LOSS CURVES ########################################

def plot_curves(history, location):

    #lets plot the train and val curve
    #get the details form the history object
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    #Train and validation accuracy
    plt.plot(epochs, acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.xlabel("Epochs")
    plt.ylabel("Accurarcy")
    #plt.title('Training and Validation accurarcy')
    plt.legend()


    #plt.savefig(location+'_accuracy.png')
    #m2k.save(location+'_accuracy.tikz')

    plt.figure()
    #Train and validation loss
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    #plt.title('Training and Validation loss')
    plt.legend()
    
    #plt.savefig(location+'_loss.png')
    #m2k.save(location+'_loss.tikz')

    plt.show()

########################################  THE END  ######################################## 