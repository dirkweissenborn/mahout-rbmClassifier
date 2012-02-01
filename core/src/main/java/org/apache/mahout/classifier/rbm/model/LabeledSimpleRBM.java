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
import org.apache.mahout.classifier.rbm.layer.SoftmaxLayer;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;

import com.google.common.io.Closeables;

/**
 * The Class LabeledSimpleRBM which is a simple rbm, but also has a softmax layer in addition to
 * the visible layer. This softmax layer can be seen as labels.
 */
public class LabeledSimpleRBM extends SimpleRBM {
	
	/** The softmax layer. */
	SoftmaxLayer softmaxLayer;
	
	/** The weight label matrix. */
	protected Matrix weightLabelMatrix; 
	
	/**
	 * Instantiates a new labeled simple rbm.
	 *
	 * @param visibleLayer the visible layer
	 * @param hiddenLayer the hidden layer
	 * @param labelLayer the label layer
	 */
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

	/**
	 * Gets the softmax layer.
	 *
	 * @return the softmax layer
	 */
	public SoftmaxLayer getSoftmaxLayer() {
		return softmaxLayer;
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.model.SimpleRBM#exciteHiddenLayer(double, boolean)
	 */
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
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.model.SimpleRBM#exciteVisibleLayer(double, boolean)
	 */
	@Override
	public void exciteVisibleLayer(double inputFactor, boolean addInput) {
		super.exciteVisibleLayer(inputFactor, addInput);
		
		if(addInput)
			softmaxLayer.addInputs(weightLabelMatrix.times(getHiddenLayer().getActivations()).times(inputFactor));		
		else
			softmaxLayer.setInputs(weightLabelMatrix.times(getHiddenLayer().getActivations()).times(inputFactor));		

		softmaxLayer.exciteNeurons();
	}
	
	/**
	 * Sets the weight label matrix.
	 *
	 * @param weightLabelMatrix the new weight label matrix
	 */
	public void setWeightLabelMatrix(Matrix weightLabelMatrix) {
		this.weightLabelMatrix = weightLabelMatrix;
	}
	
	/**
	 * Gets the weight label matrix.
	 *
	 * @return the weight label matrix
	 */
	public Matrix getWeightLabelMatrix() {
		return weightLabelMatrix;
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.model.SimpleRBM#serialize(org.apache.hadoop.fs.Path, org.apache.hadoop.conf.Configuration)
	 */
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
	
	/**
	 * Materialize.
	 *
	 * @param output the output path
	 * @param conf the hadoop config
	 * @return the labeled rbm
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
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
	
	/**
	 * Gets the label.
	 *
	 * @return the label
	 */
	public Integer getLabel() {
		return softmaxLayer.getActiveNeuron();
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.model.RBMModel#updateVisibleLayer()
	 */
	@Override
	public void updateVisibleLayer() {
		visibleLayer.updateNeurons();
		softmaxLayer.updateNeurons();
	}
	
	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.model.SimpleRBM#clone()
	 */
	@Override
	public LabeledSimpleRBM clone() {
		LabeledSimpleRBM rbm = new LabeledSimpleRBM(visibleLayer.clone(), hiddenLayer.clone(), softmaxLayer.clone());
		rbm.weightMatrix = weightMatrix.clone();
		rbm.weightLabelMatrix = weightLabelMatrix.clone();
		return rbm;
	}

}
