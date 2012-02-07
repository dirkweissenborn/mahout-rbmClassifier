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

import org.apache.mahout.classifier.rbm.model.RBMModel;
import org.apache.mahout.classifier.rbm.model.SimpleRBM;
import org.apache.mahout.classifier.rbm.network.DeepBoltzmannMachine;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * The Class BackPropTrainer is a wrapper for the backpropagation training algorithm.
 */
public class BackPropTrainer {
	
	/** The learning rate. */
	double learningRate;
	
	/**
	 * Instantiates a new back prop trainer.
	 *
	 * @param learningrate the learningrate
	 */
	public BackPropTrainer(double learningrate) {
		this.learningRate = learningrate;
	}
	
	/**
	 * Calculate weight updates.
	 *
	 * @param dbm the dbm
	 * @param input the input
	 * @param output the output
	 * @return the matrix[]
	 */
	public Matrix[] calculateWeightUpdates(DeepBoltzmannMachine dbm, Vector input, Vector output) {
		//excite and update all layers of the multilayer feedforward network
		dbm.getRBM(0).getVisibleLayer().setActivations(input);
		RBMModel currentRBM =null;
		for(int i = 0; i< dbm.getRbmCount(); i++) {
			currentRBM = dbm.getRBM(i);
			currentRBM.exciteHiddenLayer(1, false);
			currentRBM.getHiddenLayer().setProbabilitiesAsActivation();
		}
		
		//compute output layers errors
		currentRBM.getHiddenLayer().computeNeuronErrors(output);
		
		//compute errors of the other layers
		for(int i = dbm.getRbmCount()-1; i>0;i--) {
			currentRBM = dbm.getRBM(i);
			currentRBM.getVisibleLayer().
				computeNeuronErrors(currentRBM.getHiddenLayer(), 
									((SimpleRBM)currentRBM).getWeightMatrix());
		}
		
		//put the results together and compute the weightupdates
		Matrix[] result = new Matrix[dbm.getRbmCount()];
		Matrix currentMatrix;
		Vector errors, activations;
		for (int i = 0; i < result.length; i++) {
			currentRBM = dbm.getRBM(i);
			currentMatrix = ((SimpleRBM)currentRBM).getWeightMatrix();
			result[i] = new DenseMatrix(currentMatrix.rowSize(), currentMatrix.columnSize());
			
			errors = currentRBM.getHiddenLayer().getErrors();
			activations = currentRBM.getVisibleLayer().getActivations();
			for (int j = 0; j < currentMatrix.rowSize(); j++) 
				for(int k = 0; k < currentMatrix.columnSize(); k++){				
					result[i].set(j,k, 
							errors.get(k)* activations.get(j)*learningRate);
				}
			
		}	
		
		return result;
	}
	
}
