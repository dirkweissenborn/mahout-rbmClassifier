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

import org.apache.mahout.classifier.rbm.model.LabeledSimpleRBM;
import org.apache.mahout.classifier.rbm.model.SimpleRBM;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

/**
 * The Class CDTrainer is a wrapper for the contrastive divergence training algorithm.
 */
public class CDTrainer {
	
	/** The nr gibbs sampling. */
	int nrGibbsSampling;
	
	/** The learning rate. */
	double learningRate;
	
	/**
	 * Instantiates a new CD trainer.
	 *
	 * @param learningRate the learning rate
	 * @param nrGibbsSampling the nr of gibbs samplings used
	 */
	public CDTrainer(double learningRate, int nrGibbsSampling) {
		this.learningRate = learningRate;
		this.nrGibbsSampling = (nrGibbsSampling>1)?nrGibbsSampling:1;
	}
	
	/**
	 * Calculate weight updates.
	 *
	 * @param rbm the rbm
	 * @param doubledTopDown true, if weights should be doubled from hidden to visible layer
	 * @param doubledBottomUp true, if weights should be doubled from visible to hidden layer
	 * @return the matrix
	 */
	public Matrix calculateWeightUpdates(SimpleRBM rbm, boolean doubledTopDown, boolean doubledBottomUp) {
		Integer factorTopDown = (doubledTopDown)?2:1;
		Integer factorBottomUp = (doubledBottomUp)?2:1;
		
		Matrix updates = rbm.getWeightMatrix().clone();
		Matrix updatesReconstruction = rbm.getWeightMatrix().clone();
		
		//data driven updates
		rbm.exciteHiddenLayer(factorBottomUp, false);
		rbm.getHiddenLayer().updateNeurons();
		
		for(int i=0; i<updates.rowSize();i++)
			for(int j=0; j<updates.columnSize();j++)
				updates.set(i, j, 
						rbm.getHiddenLayer().getActivations().get(j)*
						rbm.getVisibleLayer().getActivations().get(i)*
						learningRate);
		
		for(int i=0; i<nrGibbsSampling; i++) {
			//reconstruction driven updates
			rbm.exciteVisibleLayer(factorTopDown, false);
			rbm.getVisibleLayer().setProbabilitiesAsActivation();
			rbm.exciteHiddenLayer(factorBottomUp, false);
			if(i<nrGibbsSampling-1)
				rbm.getHiddenLayer().updateNeurons();
			else
				rbm.getHiddenLayer().setProbabilitiesAsActivation();
		}
		
		for(int i=0; i<updatesReconstruction.rowSize();i++)
			for(int j=0; j<updates.columnSize();j++)
				updatesReconstruction.set(i, j,  
					rbm.getHiddenLayer().getExcitations().get(j)*
							rbm.getVisibleLayer().getExcitations().get(i)*
							learningRate);

		return updates.minus(updatesReconstruction);
	}
	
	/**
	 * Calculate weight updates for labeled rbms.
	 *
	 * @param rbm the rbm
	 * @param doubledTopDown the doubled top down
	 * @param doubledBottomUp the doubled bottom up
	 * @return the matrix
	 */
	public Matrix calculateWeightUpdates(LabeledSimpleRBM rbm, boolean doubledTopDown, boolean doubledBottomUp) {
		Integer factorTopDown = (doubledTopDown)?2:1;
		Integer factorBottomUp = (doubledBottomUp)?2:1;
		
		int visNeuronCount = rbm.getVisibleLayer().getNeuronCount();
		Matrix updates = new DenseMatrix(visNeuronCount+
										 rbm.getSoftmaxLayer().getNeuronCount(), 
										 rbm.getHiddenLayer().getNeuronCount());
		Matrix updatesReconstruction = updates.clone();
		
		//data driven updates
		rbm.exciteHiddenLayer(factorBottomUp, false);
		rbm.getHiddenLayer().updateNeurons();
		
		for(int i=0; i<visNeuronCount;i++)
			for(int j=0; j<updates.columnSize();j++)
				updates.set(i, j, 
						rbm.getHiddenLayer().getActivations().get(j)*
						rbm.getVisibleLayer().getActivations().get(i)*
						learningRate);
		
		for(int i=visNeuronCount; i<updates.rowSize();i++)
			for(int j=0; j<updates.columnSize();j++)
				updates.set(i, j,  
						rbm.getHiddenLayer().getActivations().get(j)*
						rbm.getSoftmaxLayer().getActivations().get(i-visNeuronCount)*
						learningRate);
	
		for(int i=0; i<nrGibbsSampling; i++) {
			//reconstruction driven updates
			rbm.exciteVisibleLayer(factorTopDown, false);
			rbm.getVisibleLayer().setProbabilitiesAsActivation();
			rbm.getSoftmaxLayer().setProbabilitiesAsActivation();
			
			rbm.exciteHiddenLayer(factorBottomUp, false);
			if(i<nrGibbsSampling-1)
				rbm.getHiddenLayer().updateNeurons();
			else
				rbm.getHiddenLayer().setProbabilitiesAsActivation();
			
		}
		
		for(int i=0; i<visNeuronCount;i++)
			for(int j=0; j<updates.columnSize();j++)
				updatesReconstruction.set(i, j,  
					rbm.getHiddenLayer().getExcitations().get(j) *
					rbm.getVisibleLayer().getExcitations().get(i)*
					learningRate);
		
		for(int i=visNeuronCount; i<updates.rowSize();i++)
			for(int j=0; j<updates.columnSize();j++)
				updatesReconstruction.set(i, j,  
						rbm.getHiddenLayer().getExcitations().get(j)*
							rbm.getSoftmaxLayer().getExcitations().get(i-visNeuronCount)*
							learningRate);


		return updates.minus(updatesReconstruction);
	}

}
