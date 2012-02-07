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
import org.apache.mahout.classifier.rbm.model.LabeledSimpleRBM;
import org.apache.mahout.classifier.rbm.model.SimpleRBM;
import org.apache.mahout.classifier.rbm.network.DeepBoltzmannMachine;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * The Class RBMGreedyPreTrainingMapper.
 */
public class RBMGreedyPreTrainingMapper extends Mapper<IntWritable, VectorWritable, IntWritable, MatrixWritable> {
	 
 	/**
 	 * The Enum BATCH.
 	 */
 	static enum BATCH { 
		 /** The SIZE of the batch. */
		 SIZE }
	
	/** The dbm. */
	DeepBoltzmannMachine dbm;
	
	/** The learning rate. */
	double learningRate;
	
	/** The label. */
	private Vector label;
	
	/** The nr. */
	private int nr;
	
	/** The nr gibbs sampling. */
	private int nrGibbsSampling;
	
	/* (non-Javadoc)
	 * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
	 */
	protected void setup(Context context) throws java.io.IOException ,InterruptedException {
		Configuration conf = context.getConfiguration();
		Path p = HadoopUtil.cachedFile(conf);
		dbm = DeepBoltzmannMachine.materialize(p, conf);
		learningRate = Double.parseDouble(conf.get("learningrate"));
		nr = Integer.parseInt(conf.get("rbmNr"));
		nrGibbsSampling = Integer.parseInt(conf.get("nrGibbsSampling"));
		Integer count = Integer.parseInt(conf.get("labelcount"));
		label = new RandomAccessSparseVector(count);		
	};
		
	/* (non-Javadoc)
	 * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
	 */
	protected void map(IntWritable key, VectorWritable value, Context context) throws java.io.IOException ,InterruptedException {
		CDTrainer trainer = new CDTrainer(learningRate, nrGibbsSampling);
				
		label.set(key.get(), 1);
		
		dbm.getRBM(0).getVisibleLayer().setActivations(value.get());
		for(int i = 0; i<nr; i++){
			//double the bottom up connection for initialization
			dbm.getRBM(i).exciteHiddenLayer(2, false);
			if(i==nr-1)
				//probabilities as activation for the data the rbm should train on
				dbm.getRBM(i).getHiddenLayer().setProbabilitiesAsActivation();
			else
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
