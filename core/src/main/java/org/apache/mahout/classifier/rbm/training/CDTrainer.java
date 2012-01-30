package org.apache.mahout.classifier.rbm.training;

import org.apache.mahout.classifier.rbm.model.LabeledSimpleRBM;
import org.apache.mahout.classifier.rbm.model.SimpleRBM;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;


public class CDTrainer {
	int nrGibbsSampling;
	double learningRate;
	
	public CDTrainer(double learningRate, int nrGibbsSampling) {
		this.learningRate = learningRate;
		this.nrGibbsSampling = (nrGibbsSampling>1)?nrGibbsSampling:1;
	}
	
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
