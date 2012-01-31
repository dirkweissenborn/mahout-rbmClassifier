package org.apache.mahout.classifier.rbm.model;

import java.io.IOException;
import java.io.StringWriter;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.rbm.layer.Layer;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.io.Closeables;

public class SimpleRBM extends RBMModel {
	protected Matrix weightMatrix; //rownumber is visible unit, column is hidden

	public SimpleRBM(Layer visibleLayer, Layer hiddenLayer) {
		this.visibleLayer = visibleLayer;
		this.hiddenLayer = hiddenLayer;
		
	    // initialize the random number generator
		Random rand = new Random();

	    this.weightMatrix = new DenseMatrix(visibleLayer.getNeuronCount(), hiddenLayer.getNeuronCount());
	    //small random values chosen from a zero-mean Gaussian with
	    //a standard deviation of about 0.01
	    for (int i = 0; i < weightMatrix.columnSize(); i++) {
	    	for (int j = 0; j < weightMatrix.rowSize(); j++) {
				weightMatrix.set(j, i, rand.nextGaussian()/100);
			}
		}
	}
	
	public SimpleRBM(Layer visibleLayer, Layer hiddenLayer, Matrix weightMatrix) {
		this.visibleLayer = visibleLayer;
		this.hiddenLayer = hiddenLayer;
		
		Random rand;
	    // initialize the random number generator
	    rand = RandomUtils.getRandom();

	    this.weightMatrix = weightMatrix;
	}

	public Matrix getWeightMatrix() {
		return weightMatrix;
	}
	
	@Override
	public void serialize(Path output, Configuration conf) throws IOException {
		FileSystem fs = output.getFileSystem(conf);
		String rbmnr = conf.get("rbmnr");
		FSDataOutputStream out = null;
		if(rbmnr!=null)
			out = fs.create(new Path(output, conf.get("rbmnr")), true);	
		else
			out = fs.create(output, true);
		
	    try {	     
	      out.writeChars(this.getClass().getName()+" ");
	      out.writeChars(visibleLayer.getClass().getName()+" ");
	      out.writeChars(hiddenLayer.getClass().getName()+" ");
	      
	      MatrixWritable.writeMatrix(out, weightMatrix);
	    } finally {
	      Closeables.closeQuietly(out);
	    }		
	}
	
	public static SimpleRBM materialize(Path output,Configuration conf) throws IOException {
	    FileSystem fs = output.getFileSystem(conf);
	    Matrix weightMatrix;
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
	    } finally {
	      Closeables.closeQuietly(in);
	    }
	    Layer vl = ClassUtils.instantiateAs(visLayerType, Layer.class,new Class[]{int.class},new Object[]{weightMatrix.rowSize()});
	    Layer hl = ClassUtils.instantiateAs(hidLayerType, Layer.class,new Class[]{int.class},new Object[]{weightMatrix.columnSize()});

	    SimpleRBM rbm = new SimpleRBM(vl, hl);
	    rbm.setWeightMatrix(weightMatrix);
	    return rbm;
	  }
		
	public void exciteHiddenLayer(double inputFactor, boolean addInput) {
		Matrix activations = visibleLayer.getTransposedActivations();

		Matrix input = activations.times(weightMatrix).times(inputFactor);
		
		if(addInput) 
			hiddenLayer.addInputs(input.viewRow(0));
		else
			hiddenLayer.setInputs(input.viewRow(0));
		
		hiddenLayer.exciteNeurons();
	}
 
	public void exciteVisibleLayer(double inputFactor, boolean addInput) {
		Vector input = weightMatrix.times(getHiddenLayer().getActivations()).times(inputFactor);
		
		if(addInput)
			visibleLayer.addInputs(input);
		else
			visibleLayer.setInputs(input);
		
		visibleLayer.exciteNeurons();
	}
	
	public void setWeightMatrix(Matrix weightMatrix) {
		this.weightMatrix = weightMatrix;
	}
	
	@Override
	public RBMModel clone() {
		SimpleRBM rbm = new SimpleRBM(visibleLayer.clone(), hiddenLayer.clone());
		rbm.weightMatrix = weightMatrix.clone();
		return rbm;
	}
}
