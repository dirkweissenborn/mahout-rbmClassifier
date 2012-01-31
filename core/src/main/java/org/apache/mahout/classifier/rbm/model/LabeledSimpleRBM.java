package org.apache.mahout.classifier.rbm.model;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.rbm.layer.Layer;
import org.apache.mahout.classifier.rbm.layer.SoftmaxLayer;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;

import com.google.common.io.Closeables;

public class LabeledSimpleRBM extends SimpleRBM {
	SoftmaxLayer softmaxLayer;
	protected Matrix weightLabelMatrix; 
	
	public LabeledSimpleRBM(Layer visibleLayer, Layer hiddenLayer, SoftmaxLayer labelLayer) {
		super(visibleLayer, hiddenLayer);
		this.softmaxLayer = labelLayer;
		
	    // initialize the random number generator
	    Random rand = new Random();
		
		this.weightLabelMatrix = new DenseMatrix(softmaxLayer.getNeuronCount(), hiddenLayer.getNeuronCount());
	    //small random values chosen from a zero-mean Gaussian with
	    //a standard deviation of about 0.01
	    for (int i = 0; i < weightLabelMatrix.columnSize(); i++) {
	    	for (int j = 0; j < weightLabelMatrix.rowSize(); j++) {
	    		weightLabelMatrix.set(j, i, rand.nextGaussian()/100);
			}
		}
	}

	public SoftmaxLayer getSoftmaxLayer() {
		return softmaxLayer;
	}
	
	@Override
	public void exciteHiddenLayer(double inputFactor, boolean addInput) {		
		Matrix activations = visibleLayer.getTransposedActivations();
		Matrix softMaxActivations = softmaxLayer.getTransposedActivations();
		
		Vector input = activations.times(weightMatrix).times(inputFactor).viewRow(0).plus(
				softMaxActivations.times(weightLabelMatrix).times(inputFactor).viewRow(0) );
		
		if(addInput) 
			hiddenLayer.addInputs(input);
		else
			hiddenLayer.setInputs(input);
		
		hiddenLayer.exciteNeurons();
	}
	
	@Override
	public void exciteVisibleLayer(double inputFactor, boolean addInput) {
		super.exciteVisibleLayer(inputFactor, addInput);
		
		if(addInput)
			softmaxLayer.addInputs(weightLabelMatrix.times(getHiddenLayer().getActivations()).times(inputFactor));		
		else
			softmaxLayer.setInputs(weightLabelMatrix.times(getHiddenLayer().getActivations()).times(inputFactor));		

		softmaxLayer.exciteNeurons();
	}
	
	public void setWeightLabelMatrix(Matrix weightLabelMatrix) {
		this.weightLabelMatrix = weightLabelMatrix;
	}
	
	public Matrix getWeightLabelMatrix() {
		return weightLabelMatrix;
	}
	
	@Override
	public void serialize(Path output, Configuration conf) throws IOException {
		FileSystem fs = output.getFileSystem(conf);		
	    FSDataOutputStream out = fs.create(new Path(output, conf.get("rbmnr")), true);
	    try {	     
	      out.writeChars(this.getClass().getName()+" ");
	      out.writeChars(visibleLayer.getClass().getName()+" ");
	      out.writeChars(hiddenLayer.getClass().getName()+" ");
	      
	      MatrixWritable.writeMatrix(out, weightMatrix);
	      MatrixWritable.writeMatrix(out, weightLabelMatrix);

	    } finally {
	      Closeables.closeQuietly(out);
	    }		
	}
	
	public static LabeledSimpleRBM materialize(Path output,Configuration conf) throws IOException {
	    FileSystem fs = output.getFileSystem(conf);
	    Matrix weightMatrix;
	    Matrix weightLabelMatrix;
	    String visLayerType = "";
	    String hidLayerType = "";
	    FSDataInputStream in = fs.open(output);
	    try {
	      
	      char chr;
	      while((chr=in.readChar())!=' ');
	      while((chr=in.readChar())!=' ')
	    	  visLayerType += chr;
	      while((chr=in.readChar())!=' ')
	    	  hidLayerType += chr;
	      
	      weightMatrix = MatrixWritable.readMatrix(in);
	      weightLabelMatrix =MatrixWritable.readMatrix(in);
	    } finally {
	      Closeables.closeQuietly(in);
	    }
	    Layer vl = ClassUtils.instantiateAs(visLayerType, Layer.class,new Class[]{int.class},new Object[]{weightMatrix.rowSize()});
	    Layer hl = ClassUtils.instantiateAs(hidLayerType, Layer.class,new Class[]{int.class},new Object[]{weightMatrix.columnSize()});

	    LabeledSimpleRBM rbm = new LabeledSimpleRBM(vl, hl, new SoftmaxLayer(weightLabelMatrix.rowSize()));
	    rbm.setWeightMatrix(weightMatrix);
	    rbm.setWeightLabelMatrix(weightLabelMatrix);
	    return rbm;
	  }
	
	public Integer getLabel() {
		return softmaxLayer.getActiveNeuron();
	}
	
	@Override
	public void updateVisibleLayer() {
		visibleLayer.updateNeurons();
		softmaxLayer.updateNeurons();
	}
	
	@Override
	public LabeledSimpleRBM clone() {
		LabeledSimpleRBM rbm = new LabeledSimpleRBM(visibleLayer.clone(), hiddenLayer.clone(), softmaxLayer.clone());
		rbm.weightMatrix = weightMatrix.clone();
		rbm.weightLabelMatrix = weightLabelMatrix.clone();
		return rbm;
	}

}
