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
package org.apache.mahout.classifier.rbm.model;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.rbm.layer.Layer;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;

import com.google.common.io.Closeables;

/**
 * The Class SimpleRBM consisting of a visible and hidden layer and a weight matrix that connects them.
 */
public class SimpleRBM extends RBMModel {
	
	/** The weight matrix. */
	protected Matrix weightMatrix; //rownumber is visible unit, column is hidden

	/**
	 * Instantiates a new simple rbm.
	 *
	 * @param visibleLayer the visible layer
	 * @param hiddenLayer the hidden layer
	 */
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
	
	/**
	 * Instantiates a new simple rbm.
	 *
	 * @param visibleLayer the visible layer
	 * @param hiddenLayer the hidden layer
	 * @param weightMatrix the weight matrix
	 */
	public SimpleRBM(Layer visibleLayer, Layer hiddenLayer, Matrix weightMatrix) {
		this.visibleLayer = visibleLayer;
		this.hiddenLayer = hiddenLayer;
	    this.weightMatrix = weightMatrix;
	}

	/**
	 * Gets the weight matrix.
	 *
	 * @return the weight matrix
	 */
	public Matrix getWeightMatrix() {
		return weightMatrix;
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.model.RBMModel#serialize(org.apache.hadoop.fs.Path, org.apache.hadoop.conf.Configuration)
	 */
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
	
	/**
	 * Materialize.
	 *
	 * @param output path to serialize to
	 * @param conf the hadoop config
	 * @return the rbm
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
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
		
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.model.RBMModel#exciteHiddenLayer(double, boolean)
	 */
	public void exciteHiddenLayer(double inputFactor, boolean addInput) {
		Matrix activations = visibleLayer.getTransposedActivations();

		Matrix input = activations.times(weightMatrix).times(inputFactor);
		
		if(addInput) 
			hiddenLayer.addInputs(input.viewRow(0));
		else
			hiddenLayer.setInputs(input.viewRow(0));
		
		hiddenLayer.exciteNeurons();
	}
 
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.model.RBMModel#exciteVisibleLayer(double, boolean)
	 */
	public void exciteVisibleLayer(double inputFactor, boolean addInput) {
		Vector input = weightMatrix.times(getHiddenLayer().getActivations()).times(inputFactor);
		
		if(addInput)
			visibleLayer.addInputs(input);
		else
			visibleLayer.setInputs(input);
		
		visibleLayer.exciteNeurons();
	}
	
	/**
	 * Sets the weight matrix.
	 *
	 * @param weightMatrix the new weight matrix
	 */
	public void setWeightMatrix(Matrix weightMatrix) {
		this.weightMatrix = weightMatrix;
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.model.RBMModel#clone()
	 */
	@Override
	public RBMModel clone() {
		SimpleRBM rbm = new SimpleRBM(visibleLayer.clone(), hiddenLayer.clone());
		rbm.weightMatrix = weightMatrix.clone();
		return rbm;
	}
}
