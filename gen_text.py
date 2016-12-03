import sys
import numpy
from keras.models import load_model

def generate_text(model, X_test, n_chars_test):

	pattern = X_test[0]
	generated = ''

	# generate characters
	for i in range(n_chars_test):
		x = numpy.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_vocab)

		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		result = int_to_char[index]
		generated = generated + result

		seq_in = [int_to_char[value] for value in pattern]
		sys.stdout.write(result)
		pattern.append(index)
		pattern = pattern[1:len(pattern)]

	print generated

	
	with open("generated_text.txt", "w") as text_file:
		text_file.write("{0}".format(generated))
	
	print "\nDone."


if __name__ == '__main__':

	model = load_model('final_model_2x256.h5')

	seq_length = 100
	unknown_token = 'U'

	filename_train = "alice-in-wonderland.txt"
	train_text = open(filename_train).read()
	train_text = train_text.lower()
	train_text = [c if c not in ['\xe2','\x80','\x99','\x98','\x9d','\x9c','0','3','*','[',']','_'] \
    else unknown_token for c in train_text]

	filename_test = "chapter12.txt"
	test_text = open(filename_test).read()
	test_text = test_text.lower()
	test_text = [c if c not in ['\xe2','\x80','\x99','\x98','\x9d','\x9c','0','3','*','[',']','_'] \
    else unknown_token for c in test_text]

	# create mapping of unique chars to integers, and a reverse mapping
	chars = sorted(list(set(train_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))

	# summarize the loaded data
	n_chars = len(train_text)
	n_vocab = len(chars)
	print "Total Characters: ", n_chars
	print "Total Vocab: ", n_vocab

	n_chars_test = len(test_text)

	dataX2 = []
	dataY2 = []


	for i in range(0, n_chars_test - seq_length, 1):
		seq_in2 = test_text[i:i + seq_length]
		seq_out2 = test_text[i + seq_length]
		dataX2.append([char_to_int[char] for char in seq_in2])
		dataY2.append(char_to_int[seq_out2])

	n_patterns_test = len(dataX2)
	generate_text(model, dataX2, n_chars_test)
