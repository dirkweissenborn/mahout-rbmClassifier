package org.apache.mahout.classifier.rbm.training;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.rbm.model.LabeledSimpleRBM;
import org.apache.mahout.classifier.rbm.model.SimpleRBM;
import org.apache.mahout.classifier.rbm.network.DeepBoltzmannMachine;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class RBMGreedyPreTrainingMapper extends Mapper<IntWritable, VectorWritable, IntWritable, MatrixWritable> {
	 static enum BATCH { SIZE }
	
	DeepBoltzmannMachine dbm;
	double learningRate;
	private Vector label;
	private int nr;
	
	protected void setup(Context context) throws java.io.IOException ,InterruptedException {
		Configuration conf = context.getConfiguration();
		Path p = HadoopUtil.cachedFile(conf);
		dbm = DeepBoltzmannMachine.materialize(p, conf);
		learningRate = Double.parseDouble(conf.get("learningrate"));
		nr = Integer.parseInt(conf.get("rbmNr"));		
		Integer count = Integer.parseInt(conf.get("labelcount"));
		label = new RandomAccessSparseVector(count);		
	};
		
	protected void map(IntWritable key, VectorWritable value, Context context) throws java.io.IOException ,InterruptedException {
		CDTrainer trainer = new CDTrainer(learningRate, 5);
				
		label.set(key.get(), 1);
		
		dbm.getRBM(0).getVisibleLayer().setActivations(value.get());
		for(int i = 0; i<nr; i++){
			dbm.getRBM(i).exciteHiddenLayer((i==0)? 2:1, false);
			dbm.getRBM(i).getHiddenLayer().updateNeurons();
		}
		
		context.getCounter(BATCH.SIZE).increment(1);
		
		if(nr==dbm.getRbmCount()-1) {
			((LabeledSimpleRBM)dbm.getRBM(nr)).getSoftmaxLayer().setActivations(label);
			
			Matrix updates = trainer.calculateWeightUpdates((LabeledSimpleRBM)dbm.getRBM(nr), true, false);
			context.write(new IntWritable(nr), new MatrixWritable(updates));
		}
		else {
			Matrix updates = trainer.calculateWeightUpdates((SimpleRBM)dbm.getRBM(nr), false, nr==0);
			context.write(new IntWritable(nr), new MatrixWritable(updates));
		}
	};

}
