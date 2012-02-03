package org.apache.mahout.classifier.rbm.test;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.rbm.RBMClassifier;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class TestRBMClassifierMapper extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable>{

	private RBMClassifier rbmCl;

	protected void setup(Context context) throws java.io.IOException ,InterruptedException {
		Configuration conf = context.getConfiguration();
		Path p = HadoopUtil.cachedFile(conf);
		rbmCl = RBMClassifier.materialize(p, conf);
	};
	
	protected void map(IntWritable key, VectorWritable value, Context context) throws java.io.IOException ,InterruptedException {
		Vector result;
		synchronized(rbmCl) {
			result = rbmCl.classify(value.get(),4);
		}
		context.write(key, new VectorWritable(result));		
	};
}
