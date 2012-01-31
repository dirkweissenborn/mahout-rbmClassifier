package org.apache.mahout.classifier.rbm.network;

import org.apache.mahout.classifier.rbm.layer.Layer;
import org.apache.mahout.math.Vector;

public class DBMStateIterator{


	public static void iterateUntilStableLayer(Layer layer,DeepBoltzmannMachine dbm, int leastStableIterations) {
		int counter = 0;
		int counter2 = 0;
		Vector activations = layer.getActivations().clone();
		//TODO experimental counter2<leastStableIterations*5; how many iterations should the classifier need
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
