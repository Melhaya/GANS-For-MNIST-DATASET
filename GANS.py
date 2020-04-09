from keras.datasets.mnist import load_data
from Models import *
from data_proc import *

######################################## LOAD MNIST DATASET ########################################

(trainX, trainy), (testX, testy) = load_data()

gan_x = Nomalize_data(trainX)
gan_y = ones((len(gan_x), 1))

# Visualize MNIST dataset
#Visualize_MNIST(trainX)

######################################## DEFINE MODELS ########################################

# define the discriminator model
discriminator = define_discriminator()
discriminator.summary()

# define the generator model
generator = define_generator(latent_dim = 100)
generator.summary()

# create the GAN
GAN = define_gan(generator, discriminator)
GAN.summary()

######################################## TRAIN ########################################

epochs = 101
BS = 256
Total_batches = gan_x.shape[0]//BS


for epoch in range(epochs):
	for batch in range(Total_batches):

		gan_x_batch = gan_x[batch*BS:(batch+1)*BS]
		gan_y_batch = gan_y[batch*BS:(batch+1)*BS]

		g_input = randn(100 * BS)
		g_input = g_input.reshape(BS, 100)
		fake_x = generator.predict(g_input)

		fake_y = zeros((BS, 1))

		D_x = concatenate((gan_x_batch, fake_x))
		D_y = concatenate((gan_y_batch, fake_y))

		#discriminator.trainable = True

		d_loss, _ = discriminator.train_on_batch(D_x, D_y)

		#discriminator.trainable = False

		g_input = randn(100 * BS)
		g_input = g_input.reshape(BS, 100)
		g_loss = GAN.train_on_batch(g_input, gan_y_batch)

		print('>%d, %d/%d, d=%.3f, g=%.3f' % (epoch+1, batch+1, Total_batches, d_loss, g_loss))

######################################## VISUALIZE GANS PERFORMANCE ########################################

	if (epoch%10 == 0):

		g_vis = randn(100 * BS)
		g_vis = g_vis.reshape(BS, 100)
		fake_x = generator.predict(g_vis)
		for i in range(100):
			# define subplot
			plt.subplot(10, 10, 1 + i)
			# turn off axis labels
			plt.axis('off')
			# plot single image
			plt.imshow(fake_x[i, :, :, 0], cmap='gray_r')
		plt.show()

######################################## SAVE FINAL GENERATOR MODEL ########################################

generator.save("generator_100epoch.h5")

########################################  THE END  ########################################