package org.apache.mahout.classifier.rbm.layer;


public class GaussianLayer extends AbstractLayer {
	private double standardDeviation;
	private double mean;
	
	public GaussianLayer(int neuroncount, double standardDeviation, double mean) {
		super(neuroncount);
		this.mean = mean;
		this.standardDeviation= standardDeviation;
	}

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

	@Override
	public double getActivationDerivativeOfNeuron(int i) {
		double xminusmean = biases.get(i)+inputs.get(i)-mean;
		
		return -xminusmean*Math.exp(-xminusmean*xminusmean / 
								 (2*standardDeviation*standardDeviation))/
				(Math.sqrt(2*Math.PI)*Math.pow(standardDeviation,3));
	}
	
	@Override
	public GaussianLayer clone() {
		return new GaussianLayer(activations.size(),standardDeviation,mean);
	}
	
	
}
