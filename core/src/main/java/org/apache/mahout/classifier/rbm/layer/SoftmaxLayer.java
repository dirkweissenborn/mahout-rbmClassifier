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


/**
 * The Class SoftmaxLayer.
 */
public class SoftmaxLayer extends AbstractLayer {

	/** The partition sum. */
	private double partitionSum;
	
	/**
	 * Instantiates a new softmax layer.
	 *
	 * @param neuronCount the neuron count
	 */
	public SoftmaxLayer(int neuronCount) {
		super(neuronCount);
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#exciteNeurons()
	 */
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
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.AbstractLayer#updateNeurons()
	 */
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
	
	/**
	 * Gets the active neuron.
	 *
	 * @return the active neuron
	 */
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

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#getActivationDerivativeOfNeuron(int)
	 */
	@Override
	public double getActivationDerivativeOfNeuron(int i) {
		double exp = Math.exp(inputs.get(i)+biases.get(i));
		return (partitionSum * exp-exp*exp)/(partitionSum*partitionSum);
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.AbstractLayer#clone()
	 */
	@Override
	public SoftmaxLayer clone() {
		return new SoftmaxLayer(activations.size());
	}

}
