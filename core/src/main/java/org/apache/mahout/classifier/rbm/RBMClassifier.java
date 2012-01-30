package org.apache.mahout.classifier.rbm;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.rbm.layer.LogisticLayer;
import org.apache.mahout.classifier.rbm.layer.SoftmaxLayer;
import org.apache.mahout.classifier.rbm.model.LabeledSimpleRBM;
import org.apache.mahout.classifier.rbm.model.RBMModel;
import org.apache.mahout.classifier.rbm.model.SimpleRBM;
import org.apache.mahout.classifier.rbm.network.DBMStateIterator;
import org.apache.mahout.classifier.rbm.network.DeepBoltzmannMachine;
import org.apache.mahout.math.Vector;

public class RBMClassifier extends AbstractVectorClassifier{

	private DeepBoltzmannMachine dbm;
	
	public RBMClassifier(int numCategories, int[] layers) {
		if(layers.length<2)
			return;
		
		RBMModel bottomRbm = null;
		if(layers.length>2)
			bottomRbm = new SimpleRBM(new LogisticLayer(layers[0]), new LogisticLayer(layers[1]));
		else
			bottomRbm = new LabeledSimpleRBM(new LogisticLayer(layers[0]), new LogisticLayer(layers[1]), new SoftmaxLayer(numCategories));
		
		dbm = new DeepBoltzmannMachine(bottomRbm);
		
		for(int i=1; i<layers.length-1; i++) {
			if(i<layers.length-2)
				dbm.stackRBM(new SimpleRBM(dbm.getLayer(i), new LogisticLayer(layers[i+1])));
			else
				dbm.stackRBM(new LabeledSimpleRBM(dbm.getLayer(i), new LogisticLayer(layers[i+1]), new SoftmaxLayer(numCategories)));		
		}
	}
	
	public DeepBoltzmannMachine getDbm() {
		return dbm;
	}
	
	@Override
	public int numCategories() {		
		return ((LabeledSimpleRBM)dbm.getRBM(dbm.getRbmCount()-1)).getSoftmaxLayer().getNeuronCount();
	}

	@Override
	public Vector classify(Vector instance) {		
		return classify(instance,5);
	}
	
	public Vector classify(Vector instance, int stableStatesCount) {
		dbm.getLayer(0).setActivations(instance);
		dbm.upPass();
		SoftmaxLayer layer = ((LabeledSimpleRBM)dbm.getRBM(dbm.getRbmCount()-1)).getSoftmaxLayer();
		
		DBMStateIterator.iterateUntilStableLayer(layer, dbm, stableStatesCount);
		
		Vector excitations = layer.getExcitations();
		return excitations.clone().viewPart(1, excitations.size()-1);
	}

	@Override
	public double classifyScalar(Vector instance) {
		return classify(instance).get(0);
	}
	
	public void serialize(Path output, Configuration conf) throws IOException {
		dbm.serialize(output, conf);
	}
	
	public static RBMClassifier materialize(Path output, Configuration conf) throws IOException {
		RBMClassifier cl = new RBMClassifier(0, new int[]{});
		cl.dbm = DeepBoltzmannMachine.materialize(output, conf);
		return cl;
	}
	
	public DeepBoltzmannMachine initializeMultiLayerNN() {
		DeepBoltzmannMachine ret= new DeepBoltzmannMachine(dbm.getRBM(0));
		
		int rbmCount = dbm.getRbmCount();
		for (int i = 1; i < rbmCount-1; i++) {
			ret.stackRBM(dbm.getRBM(i));
		}
		
		LabeledSimpleRBM rbm = (LabeledSimpleRBM)dbm.getRBM(rbmCount-1);
		SimpleRBM secondlastRbm = new SimpleRBM(rbm.getVisibleLayer(), rbm.getHiddenLayer(), rbm.getWeightMatrix());
		
		ret.stackRBM(secondlastRbm);
		SimpleRBM lastRbm = new SimpleRBM(rbm.getHiddenLayer(), rbm.getSoftmaxLayer(), rbm.getWeightLabelMatrix().transpose());
		ret.stackRBM(lastRbm);
		
		return ret;
	}
	
	public Vector getCurrentScores() {
		return ((LabeledSimpleRBM)dbm.getRBM(dbm.getRbmCount()-1)).getSoftmaxLayer().getExcitations();
	}

}
