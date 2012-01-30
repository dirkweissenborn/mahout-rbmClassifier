package org.apache.mahout.classifier.rbm.layer;

public class LogisticLayer extends AbstractLayer {

	public LogisticLayer(int neuroncount) {
		super(neuroncount);
	}

	@Override
	public void exciteNeurons() {
		for(int i=0; i< excitations.size(); i++) {
			excitations.set(i, 1/(1+Math.exp(-(inputs.get(i)+biases.get(i)))));
		}
	}

	@Override
	public double getActivationDerivativeOfNeuron(int i) {
		double exp = Math.exp(inputs.get(i)+biases.get(i));
		return exp/((exp+1)*(exp+1));
	}
}
