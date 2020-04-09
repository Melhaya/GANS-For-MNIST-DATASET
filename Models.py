from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Conv2DTranspose
 
######################################## STANDALONE DISCRIMINATOR MODEL ########################################

def define_discriminator(inputShape=(28,28,1), activation="relu"):
	model = Sequential()
	
	model.add(Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', input_shape=inputShape))
#	model.add(Activation(activation))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	
	model.add(Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same'))
#	model.add(Activation(activation))	
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	

	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	
	return model


######################################## STANDALONE GENERATOR MODEL ########################################

def define_generator(latent_dim):

	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	
	return model
 

######################################## GAN MODEL ########################################

def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

########################################  THE END  ######################################## 