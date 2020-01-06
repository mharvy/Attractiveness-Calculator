
#import net
import glob
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch


def train_net(img_length):

	# Get images and convert them to viable training set for network
	# Note that all images in the training set must be square
	images = glob.glob("/home/marc/Documents/Personal/attractiveness/faces/*png")
	training_set = torch.zeros(len(images), 3, img_length, img_length)

	# Currently this gets them in 3 channel 2d tensor form
	for i, image in enumerate(images):
		print(i)
		print(image)
		img = Image.open(image).convert("RGB")
		img = img.resize((img_length, img_length), Image.ANTIALIAS)
		#plt.imshow(img)
		#plt.show()
		tensor = transforms.ToTensor()(img)
		training_set[i] = tensor
		

def main():
	train_net(100)


if __name__ == "__main__":
	main()