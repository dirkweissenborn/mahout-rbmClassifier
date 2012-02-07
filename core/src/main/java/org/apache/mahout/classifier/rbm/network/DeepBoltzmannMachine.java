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
package org.apache.mahout.classifier.rbm.network;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.classifier.rbm.layer.Layer;
import org.apache.mahout.classifier.rbm.layer.SoftmaxLayer;
import org.apache.mahout.classifier.rbm.model.LabeledSimpleRBM;
import org.apache.mahout.classifier.rbm.model.RBMModel;
import org.apache.mahout.classifier.rbm.model.SimpleRBM;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;

import com.google.common.io.Closeables;

/**
 * A DeepBoltzmannMachine is a (deep belief) neural network consisting of a stack of restricted boltzmann machines.
 */
public class DeepBoltzmannMachine implements DeepBeliefNetwork, Cloneable{
	
	/** The restricted boltzmann machines where nr 0 is lowest. */
	private List<RBMModel> rbms;
	
	/**
	 * Instantiates a new deep boltzmann machine.
	 *
	 * @param lowestRBM the lowest rbm
	 */
	public DeepBoltzmannMachine(RBMModel lowestRBM) {
		rbms = new  ArrayList<RBMModel>();
		rbms.add(lowestRBM);
	}
	
	/**
	 * Put a new RBM on the stack.
	 *
	 * @param rbm the RBM
	 * @return true, if successful
	 */
	public boolean stackRBM(RBMModel rbm) {
		if(rbm.getVisibleLayer().equals(rbms.get(rbms.size()-1).getHiddenLayer())) {
			rbms.add(rbm);
			return true;
		}
		else
			return false;
	}
	
	/**
	 * Serialize to the output.
	 *
	 * @param output the output
	 * @param conf the conf
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
	public void serialize(Path output, Configuration conf) throws IOException {
		FileSystem fs = output.getFileSystem(conf);
		FSDataOutputStream out = fs.create(output, true);
		
		try {	
			new IntWritable(rbms.size()).write(out);
		    for (int i = 0; i < rbms.size(); i++) {
		    	if(i== 0)
		    		out.writeChars(rbms.get(i).getVisibleLayer().getClass().getName()+" ");		      
		    	out.writeChars(rbms.get(i).getHiddenLayer().getClass().getName()+" ");
		      
		    	if(i<rbms.size()-1)
		    	  MatrixWritable.writeMatrix(out, ((SimpleRBM)rbms.get(i)).getWeightMatrix());
		    	else {
		    	  MatrixWritable.writeMatrix(out, ((LabeledSimpleRBM)rbms.get(i)).getWeightMatrix());
		    	  MatrixWritable.writeMatrix(out, ((LabeledSimpleRBM)rbms.get(i)).getWeightLabelMatrix());
		    	}
		    }
		} finally {
		      Closeables.closeQuietly(out);
	    }		
	}
	
	/**
	 * Materialize from input path.
	 *
	 * @param input the input path
	 * @param conf the hadoop config
	 * @return the deep boltzmann machine
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
	public static DeepBoltzmannMachine materialize(Path input, Configuration conf) throws IOException {
		FileSystem fs = input.getFileSystem(conf);
	    String visLayerType = "";
	    String hidLayerType = "";
	    FSDataInputStream in = fs.open(input);
	    DeepBoltzmannMachine dbm = null;
	    
	    try {
	    	int rbmSize = in.readInt();
	    	
		    for (int i = 0; i < rbmSize; i++) {
		    	RBMModel rbm = null;	
		    	hidLayerType="";
		    	visLayerType="";
		    	char chr;
		    	if(i==0)
			    	while((chr=in.readChar())!=' ')
			    		visLayerType += chr;
		    	
		    	while((chr=in.readChar())!=' ')
		    		hidLayerType += chr;
		    	Matrix weightMatrix = MatrixWritable.readMatrix(in);
		  	   
		    	Layer vl;
		    	if(i==0)
		    		vl = ClassUtils.instantiateAs(visLayerType, Layer.class,new Class[]{int.class},new Object[]{weightMatrix.rowSize()});
		    	else
		    		vl = dbm.rbms.get(dbm.getRbmCount()-1).getHiddenLayer();
		    	Layer hl = ClassUtils.instantiateAs(hidLayerType, Layer.class,new Class[]{int.class},new Object[]{weightMatrix.columnSize()});
		    	
		    	if(i<rbmSize-1){
			  	    rbm = new SimpleRBM(vl, hl);
			  	    ((SimpleRBM)rbm).setWeightMatrix(weightMatrix);
			    }
		    	else {
		    		Matrix weightLabelMatrix =MatrixWritable.readMatrix(in);
		    		
		    		rbm = new LabeledSimpleRBM(vl, hl, new SoftmaxLayer(weightLabelMatrix.rowSize()));
		    	    ((LabeledSimpleRBM)rbm).setWeightMatrix(weightMatrix);
		    	    ((LabeledSimpleRBM)rbm).setWeightLabelMatrix(weightLabelMatrix);
			    }
		  	    
		  	    if(i==0)
		  	    	dbm = new DeepBoltzmannMachine(rbm);
		  	    else
		  	    	dbm.stackRBM(rbm);
			}
	    } finally {
	  	      Closeables.closeQuietly(in);
	  	    }
	    	
		return dbm;
	}
	
	/**
	 * Get the i-th RBM.
	 *
	 * @param i the i
	 * @return the rBM
	 */
	public RBMModel getRBM(Integer i) {
		if(i<=rbms.size())
			return rbms.get(i);
		else
			return null;
	}
	
	/**
	 * Gets the size of the rbm stack.
	 *
	 * @return the stacksize of rbms
	 */
	public int getRbmCount() {
		return rbms.size();
	}
	
	/**
	 * Gets the layer count.
	 *
	 * @return the layer count
	 */
	public int getLayerCount() {
		return rbms.size()+1;
	}

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.network.DeepBeliefNetwork#exciteLayer(int)
	 */
	@Override
	public void exciteLayer(int l) {
		boolean addInput = (l<getRbmCount());
		if(addInput) {
			RBMModel upperRbm = getRBM(l);
			upperRbm.exciteVisibleLayer(1, false);
		}
		
		if(l>0){
			RBMModel lowerRbm = getRBM(l-1);
			lowerRbm.exciteHiddenLayer(1, addInput);
		}
	}

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.network.DeepBeliefNetwork#getLayer(int)
	 */
	@Override
	public Layer getLayer(int l) {
		if(l<getRbmCount())
			return getRBM(l).getVisibleLayer();
		return getRBM(l-1).getHiddenLayer();
	}

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.network.DeepBeliefNetwork#upPass()
	 */
	@Override
	public void upPass() {
		for (int i = 0; i < getRbmCount(); i++) {
			RBMModel rbm = rbms.get(i);			
			rbm.exciteHiddenLayer((i<getRbmCount()-1)?2:1, false);
			rbm.updateHiddenLayer();
		}
	}

	/* (non-Javadoc)
	 * @see org.apache.mahout.classifier.rbm.network.DeepBeliefNetwork#updateLayer(int)
	 */
	@Override
	public void updateLayer(int l) {
		if(l<getRbmCount()){
			RBMModel rbm = getRBM(l);
			rbm.updateVisibleLayer();
		}
		else
			getRBM(l-1).updateHiddenLayer();
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#clone()
	 */
	public DeepBoltzmannMachine clone(){
		DeepBoltzmannMachine dbm = new DeepBoltzmannMachine(rbms.get(0).clone());
		for (int i = 1; i < rbms.size(); i++) {
			RBMModel clonedRbm = getRBM(i).clone();
			clonedRbm.setVisibleLayer(dbm.getRBM(i-1).getHiddenLayer());
			dbm.stackRBM(clonedRbm);
		}
		return dbm;
	}
}
