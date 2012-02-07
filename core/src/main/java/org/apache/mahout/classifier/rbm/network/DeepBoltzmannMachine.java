package org.apache.mahout.classifier.rbm.network;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
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

public class DeepBoltzmannMachine implements DeepBeliefNetwork, Cloneable{
	List<RBMModel> rbms;
	
	public DeepBoltzmannMachine(RBMModel lowestRBM) {
		rbms = new  ArrayList<RBMModel>();
		rbms.add(lowestRBM);
	}
	
	public boolean stackRBM(RBMModel rbm) {
		if(rbm.getVisibleLayer().equals(rbms.get(rbms.size()-1).getHiddenLayer())) {
			rbms.add(rbm);
			return true;
		}
		else
			return false;
	}
	
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
	
	public static DeepBoltzmannMachine materialize(Path output, Configuration conf) throws IOException {
		FileSystem fs = output.getFileSystem(conf);
	    String visLayerType = "";
	    String hidLayerType = "";
	    FSDataInputStream in = fs.open(output);
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
	
	public RBMModel getRBM(Integer i) {
		if(i<=rbms.size())
			return rbms.get(i);
		else
			return null;
	}
	
	public int getRbmCount() {
		return rbms.size();
	}
	
	public int getLayerCount() {
		return rbms.size()+1;
	}

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

	@Override
	public Layer getLayer(int l) {
		if(l<getRbmCount())
			return getRBM(l).getVisibleLayer();
		return getRBM(l-1).getHiddenLayer();
	}

	@Override
	public void upPass() {
		for (int i = 0; i < getRbmCount(); i++) {
			RBMModel rbm = rbms.get(i);			
			rbm.exciteHiddenLayer((i<getRbmCount()-1)?2:1, false);
			rbm.updateHiddenLayer();
		}
	}

	@Override
	public void updateLayer(int l) {
		if(l<getRbmCount()){
			RBMModel rbm = getRBM(l);
			rbm.updateVisibleLayer();
		}
		else
			getRBM(l-1).updateHiddenLayer();
	}

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
