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
 * The Class LogisticLayer.
 */
public class LogisticLayer extends AbstractLayer {

	/**
	 * Instantiates a new logistic layer with logistic units which have logistic activation functions.
	 *
	 * @param neuroncount the neuroncount
	 */
	public LogisticLayer(int neuroncount) {
		super(neuroncount);
	}

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#exciteNeurons()
	 */
	@Override
	public void exciteNeurons() {
		for(int i=0; i< excitations.size(); i++) {
			excitations.set(i, 1/(1+Math.exp(-(inputs.get(i)+biases.get(i)))));
		}
	}

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.Layer#getActivationDerivativeOfNeuron(int)
	 */
	@Override
	public double getActivationDerivativeOfNeuron(int i) {
		return excitations.get(i)*(1-excitations.get(i));
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.layer.AbstractLayer#clone()
	 */
	@Override
	public LogisticLayer clone() {
		return new LogisticLayer(activations.size());
	}
}
