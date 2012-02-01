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
package org.apache.mahout.classifier.rbm;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.rbm.layer.LogisticLayer;
import org.apache.mahout.classifier.rbm.layer.SoftmaxLayer;
import org.apache.mahout.classifier.rbm.model.LabeledSimpleRBM;
import org.apache.mahout.classifier.rbm.model.RBMModel;
import org.apache.mahout.classifier.rbm.model.SimpleRBM;
import org.apache.mahout.classifier.rbm.network.DBMStateIterator;
import org.apache.mahout.classifier.rbm.network.DeepBoltzmannMachine;
import org.apache.mahout.math.Vector;

/**
 * The Class RBMClassifier is the an implementation of the VectorClassifier interface based on 
 * the paper: http://www.cs.toronto.edu/~hinton/absps/dbm.pdf.
 */
public class RBMClassifier extends AbstractVectorClassifier implements Cloneable{

	/** The dbm which is the actual classifier model. */
	private DeepBoltzmannMachine dbm;
	
	/**
	 * Instantiates a new RBM classifier.
	 *
	 * @param numCategories the num categories
	 * @param layers the sizes of the layers used to initialize the DBM.
	 */
	public RBMClassifier(int numCategories, int[] layers) {
		if(layers.length<2)
			return;
		
		RBMModel bottomRbm = null;
		if(layers.length>2)
			bottomRbm = new SimpleRBM(new LogisticLayer(layers[0]), new LogisticLayer(layers[1]));
		else
			bottomRbm = new LabeledSimpleRBM(new LogisticLayer(layers[0]), new LogisticLayer(layers[1]), new SoftmaxLayer(numCategories));
		
		dbm = new DeepBoltzmannMachine(bottomRbm);
		
		for(int i=1; i<layers.length-1; i++) {
			if(i<layers.length-2)
				dbm.stackRBM(new SimpleRBM(dbm.getLayer(i), new LogisticLayer(layers[i+1])));
			else
				dbm.stackRBM(new LabeledSimpleRBM(dbm.getLayer(i), new LogisticLayer(layers[i+1]), new SoftmaxLayer(numCategories)));		
		}
	}
	
	/**
	 * Gets the dbm.
	 *
	 * @return the dbm
	 */
	public DeepBoltzmannMachine getDbm() {
		return dbm;
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.AbstractVectorClassifier#numCategories()
	 */
	@Override
	public int numCategories() {		
		return ((LabeledSimpleRBM)dbm.getRBM(dbm.getRbmCount()-1)).getSoftmaxLayer().getNeuronCount();
	}

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.AbstractVectorClassifier#classify(org.apache.mahout.math.Vector)
	 */
	@Override
	public Vector classify(Vector instance) {		
		return classify(instance,5);
	}
	
	/**
	 * Classify: sample until the dbm has sampled (gibbs sampling) n times
	 * the same output or 5 times n if dbm was not stable for n times in a sequence until then 
	 * where n is specified by the parameters.
	 *
	 * @param instance the instance
	 * @param stableStatesCount number of gibbs samplings = n
	 * @return vector of scores, same as classify(Vector instance)
	 */
	public Vector classify(Vector instance, int stableStatesCount) {
		dbm.getLayer(0).setActivations(instance);
		dbm.upPass();
		SoftmaxLayer layer = ((LabeledSimpleRBM)dbm.getRBM(dbm.getRbmCount()-1)).getSoftmaxLayer();
		
		DBMStateIterator.iterateUntilStableLayer(layer, dbm, stableStatesCount);
		
		Vector excitations = layer.getExcitations();
		return excitations.clone().viewPart(1, excitations.size()-1);
	}

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.AbstractVectorClassifier#classifyScalar(org.apache.mahout.math.Vector)
	 */
	@Override
	public double classifyScalar(Vector instance) {
		return classify(instance).get(0);
	}
	
	/**
	 * Serialize.
	 *
	 * @param output the output path of serialization
	 * @param conf the hadoop configuration
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
	public void serialize(Path output, Configuration conf) throws IOException {
		dbm.serialize(output, conf);
	}
	
	/**
	 * Materialize.
	 *
	 * @param input path to the model
	 * @param conf the hadoop configuration
	 * @return the RBM classifier
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
	public static RBMClassifier materialize(Path input, Configuration conf) throws IOException {
		RBMClassifier cl = new RBMClassifier(0, new int[]{});
		cl.dbm = DeepBoltzmannMachine.materialize(input, conf);
		return cl;
	}
	
	/**
	 * Initialize multi layer neural network for backpropagation training.
	 *
	 * @return the deep boltzmann machine, consisting of a stack of just SimpleRBMs
	 */
	public DeepBoltzmannMachine initializeMultiLayerNN() {
		DeepBoltzmannMachine ret= new DeepBoltzmannMachine(dbm.getRBM(0));
		
		int rbmCount = dbm.getRbmCount();
		for (int i = 1; i < rbmCount-1; i++) {
			ret.stackRBM(dbm.getRBM(i));
		}
		
		LabeledSimpleRBM rbm = (LabeledSimpleRBM)dbm.getRBM(rbmCount-1);
		SimpleRBM secondlastRbm = new SimpleRBM(rbm.getVisibleLayer(), rbm.getHiddenLayer(), rbm.getWeightMatrix());
		
		ret.stackRBM(secondlastRbm);
		SimpleRBM lastRbm = new SimpleRBM(rbm.getHiddenLayer(), rbm.getSoftmaxLayer(), rbm.getWeightLabelMatrix().transpose());
		ret.stackRBM(lastRbm);
		
		return ret;
	}
	
	/**
	 * Gets the current scores.
	 *
	 * @return the current scores
	 */
	public Vector getCurrentScores() {
		return ((LabeledSimpleRBM)dbm.getRBM(dbm.getRbmCount()-1)).getSoftmaxLayer().getExcitations();
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Object#clone()
	 */
	@Override
	public RBMClassifier clone() {
		RBMClassifier rbmCl = new RBMClassifier(0, new int[]{});
		rbmCl.dbm = dbm.clone();
		return rbmCl;
	}

}
