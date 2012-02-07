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
package org.apache.mahout.classifier.rbm.network;

import org.apache.mahout.classifier.rbm.layer.Layer;
import org.apache.mahout.math.Vector;

/**
 * The Class DBMStateIterator is a helper class for iterating DBM-states 
 */
public class DBMStateIterator{


	/**
	 * Sample (iterate) until the specified layer's activation is stable for at least the specified number of times in a row or until 
	 * the dbm was sampling for at most five (experimental) times this number.
	 *
	 * @param layer the layer
	 * @param dbm the deep boltzmann machine
	 * @param leastStableIterations the least number of stable iterations
	 */
	public static void iterateUntilStableLayer(Layer layer,DeepBoltzmannMachine dbm, int leastStableIterations) {
		int counter = 0;
		int counter2 = 0;
		Vector activations = layer.getActivations().clone();
		//TODO check: experimental counter2<leastStableIterations*5; how many iterations should the classifier need
		while(counter<leastStableIterations&&counter2<leastStableIterations*5) {			
			for(int i = 1; i<dbm.getLayerCount();i++)
				dbm.exciteLayer(i);
			
			for(int i = 1; i<dbm.getLayerCount();i++)
				dbm.updateLayer(i);
			
			if(activations.getDistanceSquared(layer.getActivations())==0) {
				counter++;
			}
			else {
				activations = layer.getActivations().clone();
				counter = 0;
			}
			counter2++;
		}
	}
}
