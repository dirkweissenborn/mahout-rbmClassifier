package org.apache.mahout.classifier.rbm.training;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.prefs.BackingStoreException;

import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.collections.map.HashedMap;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.classifier.rbm.RBMClassifier;
import org.apache.mahout.classifier.rbm.model.LabeledSimpleRBM;
import org.apache.mahout.classifier.rbm.model.SimpleRBM;
import org.apache.mahout.classifier.rbm.network.DeepBoltzmannMachine;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RBMClassifierTrainingJob extends AbstractJob{

	public static final String WEIGHT_UPDATES = "weightupdates";
	private static final Logger logger = LoggerFactory.getLogger(RBMClassifierTrainingJob.class);
	
	Matrix[] lastUpdate;
	RBMClassifier rbmCl=null;
	int iterations = 10;
    double learningrate;
    double momentum;
    boolean monitor;
    boolean initbiases;
    boolean greedy;
    boolean finetuning;
    Path[] batches = null;
    int labelcount;
    int nrGibbsSampling;
    int rbmNrtoTrain;
	
	public static void main(String[] args) throws Exception {
		if(args==null|| args.length==0)
			args = new String[]{
		          "--input", "$HOME/mnist/chunks440",
		          "--output", "$HOME/mnist/model",
		          "--structure", "784,500,1000",
		          "--labelcount", "10"	,
		          "--maxIter", "10",
		          "--rbmnr","0",
		          "--monitor","-ow","-nf","-nb"};
	    ToolRunner.run(new Configuration(), new RBMClassifierTrainingJob(), args);
	  }
	
	@Override
	public int run(String[] args) throws Exception {		
		addInputOption();
	    addOutputOption();
	    addOption(DefaultOptionCreator.maxIterationsOption().create());
	    addOption("structure", "s", "comma-separated list of layer sizes", false);
	    addOption("labelcount", "lc", "total count of labels existent in the training set", true);
	    addOption("learningrate", "lr", "learning rate at the beginning of training", "0.005");
	    addOption("momentum", "m", "momentum of learning at the beginning", "0.5");
	    addOption("rbmnr", "nr", "rbm to train, < 0 means train all", "-1");
	    addOption("nrgibbs", "gn", "number of gibbs sampling used in contrastive divergence", "5");
	    addOption(new DefaultOptionBuilder()
					.withLongName(DefaultOptionCreator.MAPREDUCE_METHOD)
					.withRequired(false)
					.withDescription("Run training with map/reduce")
					.withShortName("mr").create());
	    addOption(new DefaultOptionBuilder()
					.withLongName("nogreedy")
					.withRequired(false)
					.withDescription("Don't run greedy pre training")
					.withShortName("ng").create());
	    addOption(new DefaultOptionBuilder()
					.withLongName("nofinetuning")
					.withRequired(false)
					.withDescription("Don't run fine tuning at the end")
					.withShortName("nf").create());
	    addOption(new DefaultOptionBuilder()
					.withLongName("nobiases")
					.withRequired(false)
					.withDescription("Don't initialize biases")
					.withShortName("nb").create());
	    addOption(new DefaultOptionBuilder()
        			.withLongName("monitor")
        			.withRequired(false)
        			.withDescription(
        					"If present, errors can be monitored in cosole")
        			.withShortName("mon").create());
	    addOption(DefaultOptionCreator.overwriteOption().create());
	    
	    Map<String, String> parsedArgs = parseArguments(args);
	    if (parsedArgs == null) {
	      return -1;
	    }

	    Path input = getInputPath();
	    Path output = getOutputPath();
	    FileSystem fs = FileSystem.get(output.toUri(),getConf());
	    labelcount = Integer.parseInt(getOption("labelcount"));

	    boolean local = !hasOption("mapreduce");
	    monitor = hasOption("monitor");
	    initbiases = !hasOption("nobiases");
	    finetuning = !hasOption("nofinetuning");
	    greedy = !hasOption("nogreedy");
	    
	    if(fs.isFile(input))
	    	batches = new Path[]{input};
	    else {
	    	FileStatus[] stati = fs.listStatus(input);
	    	batches = new Path[stati.length];
	    	for (int i = 0; i < stati.length; i++) {
				batches[i] = stati[i].getPath();
			}	    		
	    }

	    iterations = Integer.valueOf(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    	learningrate = Double.parseDouble(getOption("learningrate"));	    
    	momentum = Double.parseDouble(getOption("momentum"));
    	rbmNrtoTrain = Integer.parseInt(getOption("rbmnr"));
    	nrGibbsSampling = Integer.parseInt(getOption("nrgibbs"));	    
	    
    	boolean initialize = hasOption(DefaultOptionCreator.OVERWRITE_OPTION)||!fs.exists(output)||fs.listStatus(output).length<=0;
	    
    	if (initialize) {
	      HadoopUtil.delete(getConf(), getOutputPath());
	      
	      String structure = getOption("structure");	      
	      if(structure==null||structure.isEmpty())
	    	  return -1;
	      
	      String[] layers = structure.split(",");	      
	      if (layers.length<2) {
			return -1;
	      }
	      
	      int[] actualLayerSizes = new int[layers.length];			
	      for (int i = 0; i < layers.length; i++) {
	    	 actualLayerSizes[i] = Integer.parseInt(layers[i]);
	      }
	      
	      rbmCl = new RBMClassifier(labelcount, actualLayerSizes);
	    } else {
	    	rbmCl = RBMClassifier.materialize(output, getConf());
	    }
	    
	    HadoopUtil.setSerializations(getConf());	    
	    lastUpdate = new Matrix[rbmCl.getDbm().getRbmCount()];
	    
	    if(initbiases) {
		    //init biases!
		    Vector biases = null;
		    int counter = 0;
		    for(Path batch : batches) {
		    	for (Pair<IntWritable, VectorWritable> record : new SequenceFileIterable<IntWritable, VectorWritable>(batch, getConf())) {
		    		if(biases==null)
		    			biases = record.getSecond().get().clone();
		    		else
		    			biases.plus(record.getSecond().get());
		    		counter++;
		    	}
		    }
		    rbmCl.getDbm().getLayer(0).setBiases(biases.divide(counter));
		    logger.info("Biases initialized");
	    }
	    
	    //greedy pre training with gradually decreasing learningrates
	    if(greedy) {
	    	//double weights if dbm was materialized, because it was halved after greedy pretraining
	    	if(!initialize)
		    	for(int i=1; i<rbmCl.getDbm().getRbmCount()-1; i++) {
			    	SimpleRBM rbm = (SimpleRBM)rbmCl.getDbm().getRBM(i);
			    	rbm.setWeightMatrix(rbm.getWeightMatrix().times(2));
			    }
	    	
	    	if(!local)
	    		rbmCl.serialize(output, getConf());
	    	
	    	double tempLearningrate = learningrate;
	    	if(rbmNrtoTrain<0)
			   //train all rbms
			    for(int i=0; i<rbmCl.getDbm().getRbmCount(); i++) {
			    	tempLearningrate = learningrate;
					for(int b=0; b<batches.length;b++) {			
						logger.info("Greedy training of batch "+batches[b].getName()+"\nCurrent learningrate: "+tempLearningrate);
					    for (int j = 0; j < iterations; j++) {
					    	tempLearningrate -= learningrate/(iterations*batches.length+iterations);
					    	if(local) {
					    		if(!trainGreedySeq(i, batches[b], j, tempLearningrate))
					    			return -1; 
					    	}
					    	else
							    if(!trainGreedyMR(i, batches[b], j, tempLearningrate))
							    	return -1;
					    	if((iterations>4)&&(j+1)%(iterations/5)==0)
					    		logger.info(i+"-RBM: "+Math.round(((double)j+1)/iterations*100.0)+"% on batch  done!");
					    }
				    	logger.info(((double)b)/batches.length*100+"% of training is done!");

					    if(monitor) {
							double error = rbmError(batches[b], i);
							logger.info("Average reconstruction error on batch: "+error);
						}
					}
			    }
	    	else 
	    		//train just wanted rbm
	    		for(int b=0; b<batches.length;b++) {
					logger.info("Greedy training of batch "+batches[b].getName()+"\nCurrent learningrate: "+tempLearningrate);
	    			//tempLearningrate = learningrate;
				    for (int j = 0; j < iterations; j++) {
				    		tempLearningrate -= learningrate/(iterations*batches.length+iterations);
				    		logger.info("learningrate: "+tempLearningrate);
				    	if(local) {
				    		if(!trainGreedySeq(rbmNrtoTrain, batches[b], j,tempLearningrate))
						    	return -1; 
				    	}
				    	else
						    if(!trainGreedyMR(rbmNrtoTrain, batches[b], j,tempLearningrate))
						    	return -1;				
				    	if((iterations>4)&&(j+1)%(iterations/5)==0)
				    		logger.info(rbmNrtoTrain+"-RBM: "+Math.round(((double)j+1)/iterations*100.0)+"% on batch "+batches[b].getName()+" done!");
				    }
			    	logger.info(((double)b)/batches.length*100+"% of training is done!");
			    	
				    if(monitor) {
						double error = rbmError(batches[b], rbmNrtoTrain);
						logger.info("Average reconstruction error on batch: "+error);
					}
	    		}
		    
		    //weight normalization to avoid double counting
		    for(int i=1; i<rbmCl.getDbm().getRbmCount()-1; i++) {
		    	SimpleRBM rbm = (SimpleRBM)rbmCl.getDbm().getRBM(i);
		    	rbm.setWeightMatrix(rbm.getWeightMatrix().times(0.5));
		    }
		    rbmCl.serialize(output, getConf());
		    logger.info("Pretraining done");
	    }
	    
	    if(finetuning) {
	    	if(monitor) {
	    		double error = classifierError(batches[0]);
				logger.info("Classifiers average error on batch "+batches[0].getName()+
							" before finetuning: "+error);
	    	}
	    	DeepBoltzmannMachine multiLayerDbm  = null;
	    	if(local)
	    		multiLayerDbm = rbmCl.initializeMultiLayerNN();
		    //finetuning job
		    for(int b=0; b<batches.length;b++) {
				logger.info("Finetuning on batch "+batches[b].getName()+"\nCurrent learningrate: "+learningrate);
			    for (int j = 0; j < iterations; j++) {
			    	if(local) {
			    		if(!finetuneSeq(batches[b], j, multiLayerDbm))
			    			return -1;
			    	}
			    	else
			    		if(!fintuneMR(batches[b], j))
			    			return -1;		
			    	if((iterations>4)&&(j+1)%(iterations/5)==0)
			    		logger.info(rbmNrtoTrain+"-RBM: "+Math.round(((double)j+1)/iterations*100.0)+"% on batch "+batches[b].getName()+" done!");
			    }
		    	logger.info(((double)b)/batches.length*100+"% of training is done!");

			    if(monitor) {
		    		double error = classifierError(batches[0]);
					logger.info("Classifiers average error: "+error);
		    	}
		    }
		    //final serialization
		    rbmCl.serialize(output, getConf());
		    logger.info("RBM finetuning done");
	    }
	    
		return 0;
	}

	private boolean finetuneSeq(Path batch, int iteration, DeepBoltzmannMachine multiLayerDbm) {
		Vector label = new DenseVector(labelcount);
		Map<Integer, Matrix> updates = new HashMap<Integer, Matrix>();
    	int batchsize = 0;
    	
		for (Pair<IntWritable, VectorWritable> record : new SequenceFileIterable<IntWritable, VectorWritable>(batch, getConf())) {
			label.set(record.getFirst().get(), 1);
			
			BackPropTrainer trainer = new BackPropTrainer(learningrate);
			
			Matrix[] result = trainer.calculateWeightUpdates(multiLayerDbm, record.getSecond().get(), label);

			for (int i = 0; i < result.length-1; i++) {
				if(i==result.length-2) {
					Matrix weightUpdates = new DenseMatrix(result[i].rowSize()+result[i+1].columnSize(), result[i].columnSize());
					for(int j = 0; j<weightUpdates.rowSize(); j++)
						for(int k = 0; k<weightUpdates.columnSize(); k++) {
							if(j<result[i].rowSize())
								weightUpdates.set(j, k, result[i].get(j, k));
							else
								weightUpdates.set(j, k, result[i+1].get(k, j-result[i].rowSize()));
						}
						
					if(updates.containsKey(i))
						updates.put(i, weightUpdates.plus(updates.get(i)));
					else
						updates.put(i, weightUpdates);
				}
				else
					if(updates.containsKey(i))
						updates.put(i, result[i].plus(updates.get(i)));
					else
						updates.put(i, result[i]);
			}
			batchsize++;
		}
		updateRbmCl(batchsize, (iteration==0)?0:momentum, updates);

		return true;
	}

	private boolean fintuneMR(Path batch, int iteration)
			throws IOException, InterruptedException, ClassNotFoundException {
		long batchsize;
		HadoopUtil.delete(getConf(), getTempPath(WEIGHT_UPDATES));
		HadoopUtil.cacheFiles(getOutputPath(), getConf());
		
		Job trainDBM = prepareJob(batch, getTempPath(WEIGHT_UPDATES), SequenceFileInputFormat.class, 
									DBMBackPropTrainingMapper.class, IntWritable.class, MatrixWritable.class, 
								    DBMBackPropTrainingReducer.class, IntWritable.class, MatrixWritable.class, 
								    SequenceFileOutputFormat.class);
		trainDBM.getConfiguration().set("labelcount", String.valueOf(labelcount));
		trainDBM.getConfiguration().set("learningrate", String.valueOf(learningrate));

		trainDBM.setCombinerClass(DBMBackPropTrainingReducer.class);
		
		if(!trainDBM.waitForCompletion(true))
			return false;
		
		batchsize = trainDBM.getCounters().findCounter(DBMBackPropTrainingMapper.BATCHES.SIZE).getValue();
		
		changeAndSaveModel(getOutputPath(), batchsize, (iteration==0)?0:momentum);
		return true;
	}

	class greedyTrainingThread implements Callable<Matrix> {

		private DeepBoltzmannMachine dbm;
		private Vector input;
		private Vector label;
		private CDTrainer trainer;
		int rbmNr;

		public greedyTrainingThread(DeepBoltzmannMachine dbm, Vector label, Vector input, CDTrainer trainer, int rbmNr) {
			this.dbm = dbm;
			this.label = label;
			this.input = input;
			this.trainer = trainer;
			this.rbmNr = rbmNr;
		}
		
		@Override
		public Matrix call() throws Exception {
			Matrix updates = null;
			
			dbm.getRBM(0).getVisibleLayer().setActivations(input);
			for(int i = 0; i<rbmNr; i++){
				dbm.getRBM(i).exciteHiddenLayer((i==0)? 2:1, false);
				dbm.getRBM(i).getHiddenLayer().updateNeurons();
			}
						
			if(rbmNr==dbm.getRbmCount()-1) {
				((LabeledSimpleRBM)dbm.getRBM(rbmNr)).getSoftmaxLayer().setActivations(label);
				updates = trainer.calculateWeightUpdates((LabeledSimpleRBM)dbm.getRBM(rbmNr), true, false);
			}
			else {
				updates = trainer.calculateWeightUpdates((SimpleRBM)dbm.getRBM(rbmNr), false, rbmNr==0);
			}
			return updates;
		}
		
	}
	
	private boolean trainGreedySeq(int rbmNr, Path batch, int iteration, double learningrate) throws InterruptedException, ExecutionException {
    	int batchsize = 0;
    	DeepBoltzmannMachine dbm = rbmCl.getDbm();
    	Vector label = new DenseVector(labelcount);
    	Matrix updates = null;
    	ExecutorService executor = Executors.newFixedThreadPool(20);
    	List<greedyTrainingThread> tasks = new ArrayList<RBMClassifierTrainingJob.greedyTrainingThread>();
 
		for (Pair<IntWritable, VectorWritable> record : new SequenceFileIterable<IntWritable, VectorWritable>(batch, getConf())) {
			CDTrainer trainer = new CDTrainer(learningrate, nrGibbsSampling);
			label.assign(0);
			label.set(record.getFirst().get(), 1);
			/*
			dbm.getRBM(0).getVisibleLayer().setActivations(record.getSecond().get());
			for(int i = 0; i<rbmNr; i++){
				dbm.getRBM(i).exciteHiddenLayer((i==0)? 2:1, false);
				dbm.getRBM(i).getHiddenLayer().updateNeurons();
			}
						
			if(rbmNr==dbm.getRbmCount()-1) {
				((LabeledSimpleRBM)dbm.getRBM(rbmNr)).getSoftmaxLayer().setActivations(label);
				if(updates == null)
					updates = trainer.calculateWeightUpdates((LabeledSimpleRBM)dbm.getRBM(rbmNr), true, false);
				else	
					updates=updates.plus(
						trainer.calculateWeightUpdates((LabeledSimpleRBM)dbm.getRBM(rbmNr), true, false));
			}
			else {
				if(updates == null)
					updates = trainer.calculateWeightUpdates((SimpleRBM)dbm.getRBM(rbmNr), false, rbmNr==0);
				else	
					updates=updates.plus(
						trainer.calculateWeightUpdates((SimpleRBM)dbm.getRBM(rbmNr), false, rbmNr==0));
			}*/
			
			if(batchsize<20)
				tasks.add(new greedyTrainingThread(dbm.clone(), label.clone(), record.getSecond().get(), trainer, rbmNr));
			else {
				tasks.get(batchsize%20).input = record.getSecond().get();
				tasks.get(batchsize%20).label = label.clone();
			}
			
			if(batchsize%20==19) {
				List<Future<Matrix>> futureUpdates = executor.invokeAll(tasks);
				for (int i = 0; i < futureUpdates.size(); i++) {
					if(updates==null)
						updates = futureUpdates.get(i).get();
					else
						updates = updates.plus(futureUpdates.get(i).get());
				}
			}
				
    		batchsize++;
    	}
		
		if(batchsize%20!=0) {
			List<Future<Matrix>> futureUpdates = executor.invokeAll(tasks.subList(0, (batchsize-1) %20));
			for (int i = 0; i < futureUpdates.size(); i++) {
				if(updates==null)
					updates = futureUpdates.get(i).get();
				else
					updates = updates.plus(futureUpdates.get(i).get());
			}
		}
			
		
		Map<Integer,Matrix> updateMap = new HashMap<Integer,Matrix>();
		updateMap.put(rbmNr, updates);
		updateRbmCl(batchsize, (lastUpdate[rbmNr]==null)?0:momentum, updateMap);
		
		return true;
	}

	private boolean trainGreedyMR(int rbmNr, Path batch, int iteration, double learningrate)
			throws IOException, InterruptedException, ClassNotFoundException {
		long batchsize;
		HadoopUtil.delete(getConf(), getTempPath(WEIGHT_UPDATES));
		HadoopUtil.cacheFiles(getOutputPath(), getConf());
		
		Job trainRBM = prepareJob(batch, getTempPath(WEIGHT_UPDATES), SequenceFileInputFormat.class, 
									RBMGreedyPreTrainingMapper.class, IntWritable.class, MatrixWritable.class, 
								    RBMGreedyPreTrainingReducer.class, IntWritable.class, MatrixWritable.class, 
								    SequenceFileOutputFormat.class);
		trainRBM.getConfiguration().set("rbmNr", String.valueOf(rbmNr));
		trainRBM.getConfiguration().set("labelcount", String.valueOf(labelcount));
		trainRBM.getConfiguration().set("learningrate", String.valueOf(learningrate));
		trainRBM.getConfiguration().set("nrGibbsSampling", String.valueOf(nrGibbsSampling));

		trainRBM.setCombinerClass(RBMGreedyPreTrainingReducer.class);
		
		if(!trainRBM.waitForCompletion(true))
			return false;
		
		batchsize = trainRBM.getCounters().findCounter(RBMGreedyPreTrainingMapper.BATCH.SIZE).getValue();
		
		changeAndSaveModel(getOutputPath(), batchsize, (lastUpdate[rbmNr]==null)?0:momentum);
		
		return true;
	}

	private double classifierError(Path batch) {
    	double error = 0;
    	int counter = 0;
    	Vector scores;
		for (Pair<IntWritable, VectorWritable> record : new SequenceFileIterable<IntWritable, VectorWritable>(batch, getConf())) {
    		rbmCl.classify(record.getSecond().get(),3);
    		scores = rbmCl.getCurrentScores();
    		error += scores.zSum();
    		error += 1-(2*scores.get(record.getFirst().get()));
			counter++;
    	}
		error /= counter;
		return error;
	}
	
	private double rbmError(Path batch, int rbmNr) {
		DeepBoltzmannMachine dbm = rbmCl.getDbm();
		Vector label = new DenseVector(((LabeledSimpleRBM)dbm.getRBM(dbm.getRbmCount()-1)).getSoftmaxLayer().getNeuronCount());
		
		double error = 0;
    	int counter = 0;
		
		for (Pair<IntWritable, VectorWritable> record : new SequenceFileIterable<IntWritable, VectorWritable>(batch, getConf())) {
			dbm.getRBM(0).getVisibleLayer().setActivations(record.getSecond().get());
			for(int i = 0; i<rbmNr; i++){				
				dbm.getRBM(i).exciteHiddenLayer((i==0)? 2:1, false);
				dbm.getRBM(i).getHiddenLayer().updateNeurons();
			}
			if(dbm.getRBM(rbmNr) instanceof LabeledSimpleRBM) {
				label.assign(0);
				label.set(record.getFirst().get(), 1);
				((LabeledSimpleRBM)dbm.getRBM(rbmNr)).getSoftmaxLayer().setActivations(label);
			}
			error += dbm.getRBM(rbmNr).getReconstructionError();
			counter++;
    	}
		
		error/=counter;
		return error;
	}

	private void changeAndSaveModel(Path output, long batchsize, double momentum) throws IOException {
		Map<Integer,Matrix> updates = new HashMap<Integer,Matrix>();

		for (Pair<IntWritable, MatrixWritable> record : new SequenceFileDirIterable<IntWritable, MatrixWritable>(
				getTempPath(WEIGHT_UPDATES), PathType.LIST, PathFilters.partFilter(), getConf())) {
			
			if(!updates.containsKey(record.getFirst().get()))
				updates.put(record.getFirst().get(), record.getSecond().get());
			else
				updates.put(record.getFirst().get(), 
						record.getSecond().get().plus(updates.get(record.getFirst().get())));	      	  
		}	
		
		updateRbmCl(batchsize, momentum, updates);
		
		//serialization for mappers to have actual version of the dbm
		rbmCl.serialize(output, getConf());
	}

	private void updateRbmCl(long batchsize, double momentum, Map<Integer, Matrix> updates) {
		for(Integer rbmNr : updates.keySet()) {
			if(momentum>0)
				updates.put(rbmNr, (updates.get(rbmNr).divide(batchsize).times(1-momentum)).
							plus(lastUpdate[rbmNr].times(momentum)) );
			else
				updates.put(rbmNr,updates.get(rbmNr).divide(batchsize));
			
			if(rbmNr<rbmCl.getDbm().getRbmCount()-1) {
				SimpleRBM simpleRBM = (SimpleRBM)rbmCl.getDbm().getRBM(rbmNr);
				simpleRBM.setWeightMatrix(
		    		simpleRBM.getWeightMatrix().plus(updates.get(rbmNr)));
	    	
		    }else {
		    	LabeledSimpleRBM lrbm = (LabeledSimpleRBM)rbmCl.getDbm().getRBM(rbmNr);
		    	int rowSize = lrbm.getWeightMatrix().rowSize();
				Matrix weightUpdates = updates.get(rbmNr).viewPart(0, rowSize, 0, updates.get(rbmNr).columnSize());
		    	Matrix weightLabelUpdates = updates.get(rbmNr).viewPart(rowSize, updates.get(rbmNr).rowSize()-rowSize, 0, updates.get(rbmNr).columnSize());
		    	
		    	lrbm.setWeightMatrix(lrbm.getWeightMatrix().plus(weightUpdates));
		    	lrbm.setWeightLabelMatrix(lrbm.getWeightLabelMatrix().plus(weightLabelUpdates));
		    }			
			lastUpdate[rbmNr] = updates.get(rbmNr);
		}
	}
	
}
