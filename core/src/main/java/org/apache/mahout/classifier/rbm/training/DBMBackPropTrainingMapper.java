package org.apache.mahout.classifier.rbm.training;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.rbm.RBMClassifier;
import org.apache.mahout.classifier.rbm.network.DeepBoltzmannMachine;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.VectorWritable;

public class DBMBackPropTrainingMapper extends Mapper<IntWritable, VectorWritable, IntWritable, MatrixWritable>{
	static enum BATCHES { SIZE }
	
	DeepBoltzmannMachine dbm;
	double learningrate;
	private DenseVector label;

	protected void setup(org.apache.hadoop.mapreduce.Mapper<IntWritable,VectorWritable,IntWritable,MatrixWritable>.Context context) throws java.io.IOException ,InterruptedException {
		Configuration conf = context.getConfiguration();
		Path p = HadoopUtil.cachedFile(conf);
		dbm = RBMClassifier.materialize(p, conf).initializeMultiLayerNN();
		learningrate = Double.parseDouble(conf.get("learningrate"));
		
		Integer count = Integer.parseInt(conf.get("labelcount"));
		label = new DenseVector(count);
	};
	protected void map(IntWritable key, VectorWritable value, org.apache.hadoop.mapreduce.Mapper<VectorWritable,VectorWritable,IntWritable,MatrixWritable>.Context context) throws java.io.IOException ,InterruptedException {
		label.set(key.get(), 1);
		
		BackPropTrainer trainer = new BackPropTrainer(learningrate);
		
		Matrix[] result = trainer.calculateWeightUpdates(dbm, value.get(), label);
		context.getCounter(BATCHES.SIZE).increment(1);

		for (int i = 0; i < result.length-1; i++) {
			if(i==result.length-2) {
				Matrix updates = new DenseMatrix(result[i].rowSize()+result[i+1].columnSize(), result[i].columnSize());
				for(int j = 0; j<updates.rowSize(); j++)
					for(int k = 0; k<updates.columnSize(); k++) {
						if(j<result[i].rowSize())
							updates.set(j, k, result[i].get(j, k));
						else
							updates.set(j, k, result[i+1].get(k, j-result[i].rowSize()));
					}
					
				context.write(new IntWritable(i), new MatrixWritable(updates));
			}
			else
				context.write(new IntWritable(i), new MatrixWritable(result[i]));
		}
	};
}
