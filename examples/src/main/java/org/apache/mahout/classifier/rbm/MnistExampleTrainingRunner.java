package org.apache.mahout.classifier.rbm;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.classifier.rbm.training.RBMClassifierTrainingJob;

public class MnistExampleTrainingRunner {

	/**
	 * @param args
	 * @throws Exception 
	 */
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		RBMClassifierTrainingJob job = new RBMClassifierTrainingJob();
		job.setConf(new Configuration());
		String[] args1 = {
		          "--input", "/home/dirk/mnist/out",
		          "--output", "/home/dirk/mnist/model",
		          "--structure", "784,500,1000",
		          "--labelcount", "10"	,
		          "--maxIter", "10",
		          "--monitor","-ow","-seq","-nf","-nb",
		          "-nr","0"};
		
		job.run(args1);
	}

}
