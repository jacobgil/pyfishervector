#Author: Jacob Gildenblat, 2014
#License: you may use this for whatever you like 
import sys, glob, argparse
from numpy import *
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm

def dictionary(descriptors, N):
	em = cv2.EM(N)
	em.train(descriptors)
	return float32(em.getMat("means")), float32(em.getMatVector("covs")), float32(em.getMat("weights"))[0]

def image_descriptors(file):
	_ , descriptors = cv2.SIFT().detectAndCompute(cv2.cvtColor(
		cv2.imread(file), cv2.COLOR_BGR2GRAY), None)
	return descriptors

def folder_descriptors(folder):
	files = glob.glob(folder + "/*.jpg")
	print "calculating descriptos for %d images" % len(files)
	return concatenate([image_descriptors(file) for file in files])

def likelihood_moment(x, gaussians, weights, k, moment):	
	x_moment = power(float32(x), moment) if moment > 0 else float32([1])
	probabilities = map(lambda i: weights[i] * gaussians[i], range(0, len(weights)))

	ytk = probabilities[k] / sum(probabilities)
	return x_moment * ytk

def likelihood_statistics(samples, means, covs, weights):
	s0, s1,s2 = {}, {}, {}
	samples = zip(range(0, len(samples)), samples)
	gaussians = {}
	g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights)) ]
	for i,x in samples:
		gaussians[i] = {k : g[k].pdf(x) for k in range(0, len(weights) ) }

	for k in range(0, len(weights)):
		s0[k] = reduce(lambda a, (i,x): a + likelihood_moment(x, gaussians[i], weights, k, 0), samples, 0)
		s1[k] = reduce(lambda a, (i,x): a + likelihood_moment(x, gaussians[i], weights, k, 1), samples, 0)
		s2[k] = reduce(lambda a, (i,x): a + likelihood_moment(x, gaussians[i], weights, k, 2), samples, 0)
	return s0, s1, s2

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
	return float32([((s0[k] - T * w[k]) / sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
	return float32([(s1[k] - means[k] * s0[k]) / (sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
	return float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
	v = sqrt(abs(fisher_vector)) * sign(fisher_vector)
	return v / sqrt(dot(v, v))

def fisher_vector(samples, means, covs, w):
	s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)
	T = samples.shape[0]
	covs = float32([diagonal(covs[k]) for k in range(0, covs.shape[0])])
	a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
	b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
	c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
	fv = concatenate([concatenate(a), concatenate(b), concatenate(c)])
	fv = normalize(fv)
	return fv

def generate_gmm(input_folder, N):
	words = concatenate([folder_descriptors(folder) for folder in glob.glob(input_folder + '/*')]) 
	print "Training GMM of size %d.." % N
	means, covs, weights = dictionary(words, N)

	#throw away gaussians with weights that are too small:
	th = 1.0 / N
	means = float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
	covs = float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
	weights = float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])

	save("means.gmm", means)
	save("covs.gmm", covs)
	save("weights.gmm", weights)
	return means, covs, weights

def get_fisher_vectors_from_folder(folder, gmm):
	files = glob.glob(folder + "/*.jpg")
	return float32([fisher_vector(image_descriptors(file), *gmm) for file in files])

def fisher_features(folder, gmm):
	folders = glob.glob(folder + "/*")
	print folders
	features = {f : get_fisher_vectors_from_folder(f, gmm) for f in folders}
	return features

def train(folder, gmm, features):
	classifier = train(features)

	print features.values()
	X = concatenate(features.values())
	Y = concatenate([float32([i]*len(v)) for i,v in zip(range(0, len(features)), features.values())])

	clf = svm.SVC()
	clf.fit(X, Y)
	return clf

def success_rate(classifier, features):
	print "Applying the classifier..."
	X = concatenate(features.values())
	Y = concatenate([float32([i]*len(v)) for i,v in zip(range(0, len(features)), features.values())])
	print X.shape, Y.shape
	res = float(sum([a==b for a,b in zip(classifier.predict(X), Y)])) / len(Y)
	
def load_gmm(folder = ""):
	files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]
	return map(lambda file: load(file), map(lambda s : folder + "/" , files))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f' , "--file", help="Folder with images" , default='.')
    parser.add_argument("-g" , "--loadgmm" , help="Load Gmm dictionary", action = 'store_true', default = False)
    parser.add_argument('-n' , "--number", help="Number of words in dictionary" , default=5, type=int)
    args = parser.parse_args()
    return args

args = get_args()
working_folder = args.file

gmm = load_gmm(working_folder) if args.loadgmm else generate_gmm(working_folder, args.number)
fisher_features = fisher_features(working_folder, gmm)

#TBD, split the features into training and validation
classifier = train(working_folder, gmm, fisher_features)
rate = success_rate(classifier, features)
print "Success rate is %s" % str(rate)
