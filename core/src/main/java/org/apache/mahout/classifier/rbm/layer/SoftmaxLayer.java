package org.apache.mahout.classifier.rbm.layer;

import org.apache.mahout.math.Vector;

public class SoftmaxLayer extends AbstractLayer {

	private double partitionSum;
	
	public SoftmaxLayer(int neuronCount) {
		super(neuronCount);
	}
	
	@Override
	public void exciteNeurons() {
		partitionSum = 0;
		for (int i =0; i<excitations.size(); i++) {
			excitations.set(i,
					Math.exp(inputs.get(i)+biases.get(i)));
			partitionSum += excitations.get(i);
		}
		
		for (int i =0; i<excitations.size(); i++) {
			excitations.set(i,excitations.get(i)/partitionSum);
		}
	}
	
	@Override
	public void updateNeurons() {
		double tempExc = 0;
		int nMax = 0;
		for(int i=0; i<activations.size(); i++) {
			activations.set(i, 0);
			if(excitations.get(i)>tempExc) {
				nMax = i;
				tempExc = excitations.get(i);
			}
		}
		activations.set(nMax, 1);
	}
	
	public int getActiveNeuron() {
		double tempActs = 0;
		int nMax = 0;
		for(int i=0; i<activations.size(); i++) {
			activations.set(i, 0);
			if(activations.get(i)>tempActs) {
				nMax = i;
				tempActs = activations.get(i);
			}
		}
		return nMax;
	}

	@Override
	public double getActivationDerivativeOfNeuron(int i) {
		double exp = Math.exp(inputs.get(i)+biases.get(i));
		return (partitionSum * exp+exp*exp)/(partitionSum*partitionSum);
	}

}
