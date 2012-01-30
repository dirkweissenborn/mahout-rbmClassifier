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
		          "--input", "/home/dirk/mnist/out/chunk0",
		          "--output", "/home/dirk/mnist/model",
		          "--structure", "784,800,800",
		          "--labelcount", "10"	,
		          "--maxIter", "50",
		          "--momentum", "0.5",
		          "--learningrate", "0.001",
		          "--monitor","-ow","-seq"};
		
		job.run(args1);
	}

}
