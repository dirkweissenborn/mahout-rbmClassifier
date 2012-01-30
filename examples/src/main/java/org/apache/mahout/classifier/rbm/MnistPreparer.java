package org.apache.mahout.classifier.rbm;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.InputStream;
import java.io.Reader;
import java.io.ObjectInputStream.GetField;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.apache.mahout.classifier.rbm.training.RBMClassifierTrainingJob;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.vectors.io.SequenceFileVectorWriter;

public class MnistPreparer extends AbstractJob{

	public static void main(String[] args) throws Exception {
		if(args == null || args.length==0)
			args = new String[]{"--"+DefaultOptionCreator.INPUT_OPTION,"/home/dirk/mnist",
						  		"--output","/home/dirk/mnist/out",
						  		"-cnr","440"};
		
	    ToolRunner.run(new Configuration(), new MnistPreparer(), args);
	}
	

	/**
	 * only processes 44.000 images like the paper [hinton,2006] proposed
	 * (http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf)
	 */
	@Override
	public int run(String[] args) throws Exception {		
		addInputOption();
		addOutputOption();
		//chunknumber 600 gives nullpointer exception???
		addOption("chunknumber","cnr","number of chunks to be created",true);
		
		Map<String, String> parsedArgs = parseArguments(args);
	    if (parsedArgs == null) {
	      return -1;
	    }
	    
	    Path input = getInputPath();  
	    Path output = getOutputPath(); 
	    
	    FileSystem fileSystem = input.getFileSystem(getConf());	    
	    HadoopUtil.delete(getConf(), getOutputPath());
	    
	    fileSystem.mkdirs(output);
	    
	    DataInputStream dataReader = new DataInputStream(
	    		new FileInputStream(new File(new Path(input,"train-images-idx3-ubyte").toUri().getPath())));
	    DataInputStream labelReader = new DataInputStream(
	    		new FileInputStream(new File(new Path(input,"train-labels-idx1-ubyte").toUri().getPath())));
	    
    
    	labelReader.skipBytes(8);
    	dataReader.skipBytes(16);
    	int label;
    	IntWritable labelVector = new IntWritable();
    	VectorWritable imageVector = new VectorWritable(new DenseVector(28*28));

    	double[] pixels=new double[28*28];
    	
    	Integer chunks = Integer.parseInt(getOption("chunknumber"));
		
    	SequenceFile.Writer[] writer = new SequenceFile.Writer[chunks];
    	int writernr=0;
    	Integer closedwriters = 0;
    	int cntr = 0;

    	
    	//counter for the 10 ten labels, each batch should have 44000/chunks /10(labels) examplesof each label
		Integer[][] batches = new Integer[chunks][10];
		for (int i = 0; i < batches.length; i++) {
			for(int j=0; j<10; j++)
				batches[i][j]=4400/chunks;
		}
    	
    	try {
	    	while(true) {
	    		writernr =-1;
	    		label = labelReader.readUnsignedByte();
	    		labelVector.set(label);
	    		for (int i = 0; i < pixels.length; i++) {
					pixels[i]=Double.valueOf(String.valueOf(dataReader.readUnsignedByte()))/255.0;
				}
	    		for(int i = closedwriters; i<chunks; i++) {
	    			if(batches[i][label]>0) {
	    				writernr = i;
	    				//open writers only when they are needed
	    				if(writer[writernr]==null)
	    					writer[writernr] = new Writer(fileSystem, getConf(), new Path(output,"chunk"+i), IntWritable.class, VectorWritable.class);
	    				break;
	    			} else
	    				//close writers, that are opened, yet finished
	    				for(int j=0;j<10;j++) {
	    					if(batches[i][j]!=0)
	    						break;
	    					if(j==9){
	    						writer[i].close();
	    						closedwriters++;
	    					}
	    				}
	    		}
	    		
	    		if(closedwriters>=chunks)
	    			break;
	    		if(writernr==-1)
	    			continue;
	    		
	    		imageVector.get().assign(pixels);
	    		writer[writernr].append(labelVector, imageVector);
	    		cntr++;
	    		
	    		if(cntr%1000==0)
	    			Logger.getLogger(this.getClass()).info(cntr+" processed pairs");
	    		
	    		
	    		batches[writernr][label]--;
	    	}
    	}
    	catch(EOFException ex){
    		//close last writer
    		writer[writernr].close();
    	}
    	
		return 0;
	}
	
}
