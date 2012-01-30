package org.apache.mahout.classifier.rbm.layer;

import java.util.Random;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

public abstract class AbstractLayer implements Layer {
	
	protected Vector activations;
	protected Vector excitations;
	protected Vector inputs;
	protected Vector biases;
	protected Vector errors;
	
	public AbstractLayer(int neuroncount) {
		activations = new DenseVector(neuroncount);
		excitations = new DenseVector(neuroncount);
		inputs = new DenseVector(neuroncount);
		biases = new RandomAccessSparseVector(neuroncount);
		errors = new DenseVector(neuroncount);
	}
	
	@Override
	public int getNeuronCount() {
		return inputs.size();
	}

	/*@Override
	public Neuron getNeuron(int i) {
		return neurons[i];
	}*/
	
	@Override
	public void updateNeurons() {
		for (int i = 0; i<activations.size(); i++) {
			double nextDouble = new Random().nextDouble();
			activations.set(i, (nextDouble>excitations.get(i))?0:1);
		}
	}

	@Override
	public void setActivations(Vector activations) {
		if(activations.size()!=this.activations.size())
			return;
		
		this.activations = activations;
	}
	
	@Override
	public Vector getExcitations() {
		return excitations;
	}

	@Override
	public Vector getActivations() {
		return activations;
	}
	
	@Override
	public void setProbabilitiesAsActivation() {
		activations = excitations.clone();
	}
	
	@Override
	public Matrix getTransposedActivations() {
		return new DenseMatrix(1,getNeuronCount()).assignRow(0, activations);
	}
	
	@Override
	public void computeNeuronErrors(Vector output) {
		for (int i = 0; i < activations.size(); i++) {
			errors.set(i,
					getActivationDerivativeOfNeuron(i)*
					(output.get(i)-activations.get(i)));
		}
	}
	
	@Override
	public void computeNeuronErrors(Layer nextLayer, Matrix weightMatrix) {
		double sum = 0;
		Vector nextErrors = nextLayer.getErrors();
		for (int i = 0; i < activations.size(); i++) {
			sum = 0;
			for (int j = 0; j < nextLayer.getNeuronCount(); j++) {
				sum += nextErrors.get(j)*weightMatrix.get(i, j);
			}
			errors.set(i,
					getActivationDerivativeOfNeuron(i)*sum);
			
		}
	}

	@Override
	public void inputNeuron(int neuron, double input, boolean addInput) {
		if(addInput)
			inputs.set(neuron, inputs.get(neuron)+input);
		else
			inputs.set(neuron, input);
	}
	

	@Override
	public Vector getErrors() {
		return errors;
	}
	
	@Override
	public void setInputs(Vector inputs) {
		this.inputs = inputs;
	}
	
	@Override
	public void addInputs(Vector inputs) {
		this.inputs = this.inputs.plus(inputs);
	}
	
	@Override
	public void setBiases(Vector biases) {
		this.biases = biases;
	}
}
