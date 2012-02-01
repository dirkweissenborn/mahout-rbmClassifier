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
 * The Class GaussianLayer consisting of gaussian units (means they have gaussian activation functions).
 */
public class GaussianLayer extends AbstractLayer {
	
	/** The standard deviation. */
	private double standardDeviation;
	
	/** The mean. */
	private double mean;
	
	/**
	 * Instantiates a new gaussian layer.
	 *
	 * @param neuroncount the neuron count
	 * @param standardDeviation the standard deviation
	 * @param mean the mean
	 */
	public GaussianLayer(int neuroncount, double standardDeviation, double mean) {
		super(neuroncount);
		this.mean = mean;
		this.standardDeviation= standardDeviation;
	}

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#exciteNeurons()
	 */
	@Override
	public void exciteNeurons() {
		double xminusmean, excitation;
		for(int i=0; i< excitations.size(); i++) {
			xminusmean = biases.get(i)+inputs.get(i)-mean;
			excitation = 1/(Math.sqrt(2*Math.PI)*standardDeviation)*Math.exp(-xminusmean*xminusmean / 
													(2*standardDeviation*standardDeviation));
		
			excitations.set(i, excitation);
		}
	}

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#getActivationDerivativeOfNeuron(int)
	 */
	@Override
	public double getActivationDerivativeOfNeuron(int i) {
		double xminusmean = biases.get(i)+inputs.get(i)-mean;
		
		return -xminusmean*Math.exp(-xminusmean*xminusmean / 
								 (2*standardDeviation*standardDeviation))/
				(Math.sqrt(2*Math.PI)*Math.pow(standardDeviation,3));
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.AbstractLayer#clone()
	 */
	@Override
	public GaussianLayer clone() {
		return new GaussianLayer(activations.size(),standardDeviation,mean);
	}
	
	
}
