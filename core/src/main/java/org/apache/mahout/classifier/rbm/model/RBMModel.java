package org.apache.mahout.classifier.rbm.model;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

import net.sf.cglib.asm.Type;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.rbm.layer.Layer;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import com.google.common.io.Closeables;

public abstract class RBMModel {

	protected Layer visibleLayer;
	protected Layer hiddenLayer;	
	
	public Layer getVisibleLayer() {
		return visibleLayer;
	}
	
	public Layer getHiddenLayer() {
		return hiddenLayer;
	}
	
	public abstract void exciteHiddenLayer(double inputFactor, boolean addInput);
	public abstract void exciteVisibleLayer(double inputFactor, boolean addInput);
	public abstract void serialize(Path output, Configuration conf) throws IOException;
	
	public static RBMModel materialize(Path output, Configuration conf) throws IOException {
		FileSystem fs = output.getFileSystem(conf);
		String rbmType = "";
	    FSDataInputStream in = fs.open(output);

		try {
			char chr;
			while((chr=in.readChar())!=' ')
		    	  rbmType += chr;
		
	 	} finally {
	 		Closeables.closeQuietly(in);
	    }
		
		try {
			return (RBMModel)Class.forName(rbmType).
						getMethod("materialize",Path.class, Configuration.class).invoke(null,output, conf);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}
	
	public void setVisibleLayer(Layer visibleLayer) {
		this.visibleLayer = visibleLayer;
	}

	public double getReconstructionError(Vector input) {
		this.visibleLayer.setActivations(input);
		exciteHiddenLayer(1,false);
		getHiddenLayer().updateNeurons();
		exciteVisibleLayer(1,false);
		DistanceMeasure dm = new EuclideanDistanceMeasure();
		
		return dm.distance(input, visibleLayer.getExcitations());
	}

	public void updateHiddenLayer() {
		hiddenLayer.updateNeurons();
	}
	
	public void updateVisibleLayer() {
		visibleLayer.updateNeurons();
	}

	public double getReconstructionError() {
		Vector input = this.visibleLayer.getActivations().clone();
		exciteHiddenLayer(1,false);
		updateHiddenLayer();
		exciteVisibleLayer(1,false);
		
		DistanceMeasure dm = new EuclideanDistanceMeasure();		
		return dm.distance(input, visibleLayer.getExcitations());
	}
}