package org.apache.mahout.classifier.rbm.training;

import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;

public class DBMBackPropTrainingReducer extends Reducer<IntWritable,MatrixWritable,IntWritable,MatrixWritable>{

	protected void reduce(IntWritable key, Iterable<MatrixWritable> matrices, Context context) throws java.io.IOException ,InterruptedException {
		if(!matrices.iterator().hasNext())
			return;
		Matrix result = matrices.iterator().next().get();
		for (Iterator<MatrixWritable> iterator = matrices.iterator(); iterator.hasNext();) {
			 result = result.plus(iterator.next().get());
		}
		context.write(key, new MatrixWritable(result));
	};
}
