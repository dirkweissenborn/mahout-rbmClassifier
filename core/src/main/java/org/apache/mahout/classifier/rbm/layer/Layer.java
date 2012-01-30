package org.apache.mahout.classifier.rbm.layer;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

public interface Layer {

	int getNeuronCount();	
	//Neuron getNeuron(int i);
	void inputNeuron(int neuron, double input, boolean addInput);
	/**
	 * sets the total input of all neurons and calculates their excitation (probability of being active)
	 * @param inputs
	 */
	void exciteNeurons();
	void updateNeurons();
	void setActivations(Vector activations);
	void setProbabilitiesAsActivation();
	void setInputs(Vector inputs);
	void addInputs(Vector inputs);
	void setBiases(Vector biases);
	
	Vector getExcitations();
	Vector getActivations();
	Matrix getTransposedActivations();
	Vector getErrors();
	
	double getActivationDerivativeOfNeuron(int i);
	
	void computeNeuronErrors(Vector output);
	void computeNeuronErrors(Layer nextLayer, Matrix weightMatrix);
}
