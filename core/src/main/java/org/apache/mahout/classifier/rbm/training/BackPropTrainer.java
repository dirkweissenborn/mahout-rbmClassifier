package org.apache.mahout.classifier.rbm.training;

import org.apache.mahout.classifier.rbm.model.RBMModel;
import org.apache.mahout.classifier.rbm.model.SimpleRBM;
import org.apache.mahout.classifier.rbm.network.DeepBoltzmannMachine;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

public class BackPropTrainer {
	double learningRate;

	public BackPropTrainer(double learningrate) {
		this.learningRate = learningrate;
	}
	
	public Matrix[] calculateWeightUpdates(DeepBoltzmannMachine dbm, Vector input, Vector output) {
		dbm.getRBM(0).getVisibleLayer().setActivations(input);
		RBMModel currentRBM =null;
		for(int i = 0; i< dbm.getRbmCount(); i++) {
			currentRBM = dbm.getRBM(i);
			currentRBM.exciteHiddenLayer(1, false);
			currentRBM.getHiddenLayer().setProbabilitiesAsActivation();
		}
		
		currentRBM.getHiddenLayer().computeNeuronErrors(output);
		//error = currentRBM.getHiddenLayer().getErrors().zSum();
		
		for(int i = dbm.getRbmCount()-1; i>0;i--) {
			currentRBM = dbm.getRBM(i);
			currentRBM.getVisibleLayer().
				computeNeuronErrors(currentRBM.getHiddenLayer(), 
									((SimpleRBM)currentRBM).getWeightMatrix());
		}
		
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
