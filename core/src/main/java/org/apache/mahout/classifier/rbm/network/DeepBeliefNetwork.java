package org.apache.mahout.classifier.rbm.network;

import org.apache.mahout.classifier.rbm.layer.Layer;

public interface DeepBeliefNetwork {
	
	void exciteLayer(int l);
	void updateLayer(int l);
	Layer getLayer(int l);
	void upPass(); 
}
