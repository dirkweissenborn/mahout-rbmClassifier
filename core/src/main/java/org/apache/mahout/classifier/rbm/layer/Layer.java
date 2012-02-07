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

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * The Interface Layer.
 */
public interface Layer {

	/**
	 * Gets the neuron count.
	 *
	 * @return the neuron count
	 */
	int getNeuronCount();	

	/**
	 * update the input of a specified unit.
	 *
	 * @param neuron the neuron
	 * @param input the input
	 * @param addInput if true input is added to current input of unit
	 */
	void inputNeuron(int neuron, double input, boolean addInput);
	
	/**
	 * sets the total input of all neurons and calculates their excitation (probability of being active).
	 *
	 */
	void exciteNeurons();
	
	/**
	 * Update neurons, according to their excitations.
	 */
	void updateNeurons();
	
	/**
	 * Sets the activations.
	 *
	 * @param activations the new activations
	 */
	void setActivations(Vector activations);
	
	/**
	 * Sets the probabilities as activation.
	 */
	void setProbabilitiesAsActivation();
	
	/**
	 * Sets the inputs.
	 *
	 * @param inputs the new inputs
	 */
	void setInputs(Vector inputs);
	
	/**
	 * Adds the inputs.
	 *
	 * @param inputs the inputs
	 */
	void addInputs(Vector inputs);
	
	/**
	 * Sets the biases.
	 *
	 * @param biases the new biases
	 */
	void setBiases(Vector biases);
	
	/**
	 * Gets the excitations.
	 *
	 * @return the excitations
	 */
	Vector getExcitations();
	
	/**
	 * Gets the activations.
	 *
	 * @return the activations
	 */
	Vector getActivations();
	
	/**
	 * Gets the transposed activations as a matrix with one row (row vector).
	 *
	 * @return the transposed activations
	 */
	Matrix getTransposedActivations();
	
	/**
	 * Gets the errors.
	 *
	 * @return the errors
	 */
	Vector getErrors();
	
	/**
	 * Gets the activation derivative of neuron.
	 *
	 * @param i the i
	 * @return the activation derivative of neuron
	 */
	double getActivationDerivativeOfNeuron(int i);
	
	/**
	 * Compute neuron errors of the whole layer provided the should-be-output.
	 *
	 * @param output the output
	 */
	void computeNeuronErrors(Vector output);
	
	/**
	 * Compute neuron errors provided the next layer with its errors and the connections between the two layers.
	 *
	 * @param nextLayer the next layer
	 * @param weightMatrix the weight matrix
	 */
	void computeNeuronErrors(Layer nextLayer, Matrix weightMatrix);
	
	/**
	 * Clone.
	 *
	 * @return the layer
	 */
	public Layer clone();
}
