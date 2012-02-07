/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.classifier.rbm.training;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
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

/**
 * The Class DBMBackPropTrainingMapper for backpropagation training.
 */
public class DBMBackPropTrainingMapper extends Mapper<IntWritable, VectorWritable, IntWritable, MatrixWritable>{
	
	/**
	 * The Enum BATCHES.
	 */
	static enum BATCHES { 
		 /** The SIZE. */
		 SIZE 
		 }
	
	/** The dbm. */
	DeepBoltzmannMachine dbm;
	
	/** The learningrate. */
	double learningrate;
	
	/** The label. */
	private DenseVector label;

	/* (non-Javadoc)
	 * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
	 */
	protected void setup(Context context) throws java.io.IOException ,InterruptedException {
		Configuration conf = context.getConfiguration();
		Path p = HadoopUtil.cachedFile(conf);
		dbm = RBMClassifier.materialize(p, conf).initializeMultiLayerNN();
		learningrate = Double.parseDouble(conf.get("learningrate"));
		
		Integer count = Integer.parseInt(conf.get("labelcount"));
		label = new DenseVector(count);
	};
	
	/* (non-Javadoc)
	 * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
	 */
	protected void map(IntWritable key, VectorWritable value, Context context) throws java.io.IOException ,InterruptedException {
		for (int i = 0; i < label.size(); i++)
			label.setQuick(i, 0);
		label.set(key.get(), 1);
		
		BackPropTrainer trainer = new BackPropTrainer(learningrate);
		
		Matrix[] result = trainer.calculateWeightUpdates(dbm, value.get(), label);
		context.getCounter(BATCHES.SIZE).increment(1);

		//write for each RBM i (key, number of rbm) the result and put together the last two
		//matrices since they refer to just one labeled rbm, which was split to two for the training
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
