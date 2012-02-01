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
package org.apache.mahout.classifier.rbm.layer;

import java.util.Random;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * The Class AbstractLayer.
 */
public abstract class AbstractLayer implements Layer {
	
	/** The activations. */
	protected Vector activations;
	
	/** The excitations. */
	protected Vector excitations;
	
	/** The inputs. */
	protected Vector inputs;
	
	/** The biases. */
	protected Vector biases;
	
	/** The errors. */
	protected Vector errors;
	
	/**
	 * Instantiates a new abstract layer.
	 *
	 * @param neuroncount the neuroncount
	 */
	public AbstractLayer(int neuroncount) {
		activations = new DenseVector(neuroncount);
		excitations = new DenseVector(neuroncount);
		inputs = new DenseVector(neuroncount);
		biases = new RandomAccessSparseVector(neuroncount);
		errors = new DenseVector(neuroncount);
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#getNeuronCount()
	 */
	@Override
	public int getNeuronCount() {
		return inputs.size();
	}

	/*@Override
	public Neuron getNeuron(int i) {
		return neurons[i];
	}*/
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#updateNeurons()
	 */
	@Override
	public void updateNeurons() {
		for (int i = 0; i<activations.size(); i++) {
			double nextDouble = new Random().nextDouble();
			activations.set(i, (nextDouble>excitations.get(i))?0:1);
		}
	}

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#setActivations(org.apache.mahout.math.Vector)
	 */
	@Override
	public void setActivations(Vector activations) {
		if(activations.size()!=this.activations.size())
			return;
		
		this.activations = activations;
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#getExcitations()
	 */
	@Override
	public Vector getExcitations() {
		return excitations;
	}

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#getActivations()
	 */
	@Override
	public Vector getActivations() {
		return activations;
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#setProbabilitiesAsActivation()
	 */
	@Override
	public void setProbabilitiesAsActivation() {
		activations = excitations.clone();
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#getTransposedActivations()
	 */
	@Override
	public Matrix getTransposedActivations() {
		return new DenseMatrix(1,getNeuronCount()).assignRow(0, activations);
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#computeNeuronErrors(org.apache.mahout.math.Vector)
	 */
	@Override
	public void computeNeuronErrors(Vector output) {
		for (int i = 0; i < activations.size(); i++) {
			errors.set(i,
					getActivationDerivativeOfNeuron(i)*
					(output.get(i)-activations.get(i)));
		}
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#computeNeuronErrors(org.apache.mahout.classifier.rbm.layer.Layer, org.apache.mahout.math.Matrix)
	 */
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

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#inputNeuron(int, double, boolean)
	 */
	@Override
	public void inputNeuron(int neuron, double input, boolean addInput) {
		if(addInput)
			inputs.set(neuron, inputs.get(neuron)+input);
		else
			inputs.set(neuron, input);
	}
	

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#getErrors()
	 */
	@Override
	public Vector getErrors() {
		return errors;
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#setInputs(org.apache.mahout.math.Vector)
	 */
	@Override
	public void setInputs(Vector inputs) {
		this.inputs = inputs;
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#addInputs(org.apache.mahout.math.Vector)
	 */
	@Override
	public void addInputs(Vector inputs) {
		this.inputs = this.inputs.plus(inputs);
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#setBiases(org.apache.mahout.math.Vector)
	 */
	@Override
	public void setBiases(Vector biases) {
		this.biases = biases;
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Object#clone()
	 */
	@Override
	public abstract Layer clone();
}
