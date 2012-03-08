package org.apache.mahout.classifier.rbm;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.classifier.rbm.model.LabeledSimpleRBM;
import org.apache.mahout.classifier.rbm.network.DBMStateIterator;
import org.apache.mahout.classifier.rbm.network.DeepBoltzmannMachine;
import org.apache.mahout.classifier.rbm.test.TestRBMClassifierJob;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Matrix;


public class TestHardWiring extends AbstractJob{
	public static void main(String[] args) {
		new TestHardWiring().run(new String[]{"/home/dirk/models/model_440chunks_nofine",
													"/home/dirk/models/experimentalModel",
													"/home/dirk/mnist/chunks9",
													"/home/dirk/mnist/chunks9/chunk0"});
	}

	public int run(String[] args) {
		setConf(new Configuration());
		String inputPath = args[0];
		String outputPath = args[1];
		String trainingDataPath = args[2];
		String testDataPath = args[3];
		
		try {
			return transformLayer(new Path(inputPath), 
					new Path(outputPath), 
					new Path(trainingDataPath), 
					new Path(testDataPath));
		} catch (Exception e) {
			e.printStackTrace();
		}
		return -1;
	}
	
	public int transformLayer(Path input, Path output, Path trainingData, Path testData) throws Exception {
		RBMClassifier rbmCl = RBMClassifier.materialize(input, getConf());
		
		DeepBoltzmannMachine dbm = rbmCl.getDbm();
		FileSystem fs = input.getFileSystem(getConf());
		Path[] batches;
		if(fs.isFile(trainingData))
	    	batches = new Path[]{trainingData};
	    else {
	    	FileStatus[] stati = fs.listStatus(trainingData);
	    	batches = new Path[stati.length];
	    	for (int i = 0; i < stati.length; i++) {
				batches[i] = stati[i].getPath();
			}	    		
	    }
		
		Vector[] probs = new Vector[10];
		for (int i = 0; i < probs.length; i++) {
			probs[i] = new DenseVector(dbm.getLayer(dbm.getRbmCount()).getNeuronCount());
		}
		
		int[] counter = new int[10];
		int count = 0;
		for (int i = 0; i < batches.length; i++) {		
			SequenceFileIterable<IntWritable, VectorWritable> dirIterable =
			        new SequenceFileIterable<IntWritable, VectorWritable>( batches[i], getConf());
	
			for (Pair<IntWritable, VectorWritable> record : dirIterable) {
				dbm.getLayer(0).setActivations(record.getSecond().get());
				dbm.upPass();
				DBMStateIterator.iterateUntilStableLayer(dbm.getLayer(0), dbm, 3);
				probs[record.getFirst().get()]=probs[record.getFirst().get()].plus(dbm.getLayer(dbm.getRbmCount()).getActivations());
				counter[record.getFirst().get()]++;
				count++;
				if(count%1000==0)
					System.out.println(count);
			}
		}
		
		Vector total = null;
		for (int i = 0; i < counter.length; i++) {
			if(total==null)
				total=probs[i].clone();
			else
				total=total.plus(probs[i]);
			probs[i] = probs[i].divide(counter[i]);
		}
		
		total = total.divide(count);
		Vector logTotal = total.clone();
		Vector negativeLogTotal = total.clone();
		double log2= Math.log(2);
		for (int i = 0; i < total.size(); i++) {
			logTotal.set(i, Math.log(total.get(i))/log2);
			negativeLogTotal.set(i, Math.log(1-total.get(i))/log2);
		}
		LabeledSimpleRBM lrbm = (LabeledSimpleRBM)dbm.getRBM(dbm.getRbmCount()-1);

		Vector biases = new DenseVector(10);
		Matrix weights = lrbm.getWeightLabelMatrix().clone();
		
		for (int i = 0; i < probs.length; i++) {			
			Vector negInformation = probs[i].times(-1).plus(1).
									plus(total.plus(-1)).
									times(negativeLogTotal);
			Vector posInformation = probs[i].
									minus(total).
									times(logTotal);
			biases.set(i, negInformation.zSum());
			
			weights.assignRow(i, posInformation.minus(negInformation));
		}
		lrbm.getSoftmaxLayer().setBiases(biases);
		lrbm.setWeightLabelMatrix(weights);
		
		rbmCl.serialize(output, getConf());
		
		TestRBMClassifierJob tester = new TestRBMClassifierJob();
		tester.run(new String[]{"-m",output.toUri().getPath(),
								"-labelcount","10",
								"-i",testData.toUri().getPath()});
		
		return 0;
	}
}
