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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.rbm.layer.Layer;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.Vector;

import com.google.common.io.Closeables;

/**
 * The Class RBMModel.
 */
public abstract class RBMModel {

	/** The visible layer. */
	protected Layer visibleLayer;
	
	/** The hidden layer. */
	protected Layer hiddenLayer;	
	
	/**
	 * Gets the visible layer.
	 *
	 * @return the visible layer
	 */
	public Layer getVisibleLayer() {
		return visibleLayer;
	}
	
	/**
	 * Gets the hidden layer.
	 *
	 * @return the hidden layer
	 */
	public Layer getHiddenLayer() {
		return hiddenLayer;
	}
	
	/**
	 * Excite hidden layer.
	 *
	 * @param inputFactor factor which should be multiplied with the inputs of the units, usually 1
	 * @param addInput true if input should be added to the input of the hidden layer
	 */
	public abstract void exciteHiddenLayer(double inputFactor, boolean addInput);
	
	/**
	 * Excite visible layer.
	 *
	 * @param inputFactor factor which should be multiplied with the inputs of the units, usually 1
	 * @param addInput true if input should be added to the input of the hidden layer
	 */
	public abstract void exciteVisibleLayer(double inputFactor, boolean addInput);
	
	/**
	 * Serialize.
	 *
	 * @param output output path where the model should be serialized to
	 * @param conf the conf
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
	public abstract void serialize(Path output, Configuration conf) throws IOException;
	
	/**
	 * Materialize.
	 *
	 * @param input the path to the model
	 * @param conf the hadoop configuration
	 * @return the model
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
	public static RBMModel materialize(Path input, Configuration conf) throws IOException {
		FileSystem fs = input.getFileSystem(conf);
		String rbmType = "";
	    FSDataInputStream in = fs.open(input);

		try {
			char chr;
			while((chr=in.readChar())!=' ')
		    	  rbmType += chr;
		
	 	} finally {
	 		Closeables.closeQuietly(in);
	    }
		
		try {
			return (RBMModel)Class.forName(rbmType).
						getMethod("materialize",Path.class, Configuration.class).invoke(null,input, conf);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}
	
	/**
	 * Sets the visible layer.
	 *
	 * @param visibleLayer the new visible layer
	 */
	public void setVisibleLayer(Layer visibleLayer) {
		this.visibleLayer = visibleLayer;
	}

	/**
	 * Gets the reconstruction error by doing one time gibbs sampling and computing the euclidean distance
	 * between the input and the reconstruction.
	 *
	 * @param input the input
	 * @return the reconstruction error
	 */
	public double getReconstructionError(Vector input) {
		this.visibleLayer.setActivations(input);
		exciteHiddenLayer(1,false);
		getHiddenLayer().updateNeurons();
		exciteVisibleLayer(1,false);
		DistanceMeasure dm = new EuclideanDistanceMeasure();
		
		return dm.distance(input, visibleLayer.getExcitations());
	}

	/**
	 * Update hidden layer according to their excitations.
	 */
	public void updateHiddenLayer() {
		hiddenLayer.updateNeurons();
	}
	
	/**
	 * Update visible layer.according to their excitations
	 */
	public void updateVisibleLayer() {
		visibleLayer.updateNeurons();
	}

	/**
	 * Gets the reconstruction error of the current visible layers data.
	 *
	 * @return the reconstruction error
	 */
	public double getReconstructionError() {
		Vector input = this.visibleLayer.getActivations().clone();
		exciteHiddenLayer(1,false);
		updateHiddenLayer();
		exciteVisibleLayer(1,false);
		
		DistanceMeasure dm = new EuclideanDistanceMeasure();		
		return dm.distance(input, visibleLayer.getExcitations());
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Object#clone()
	 */
	public abstract RBMModel clone();
}