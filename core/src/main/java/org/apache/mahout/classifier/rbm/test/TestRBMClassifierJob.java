package org.apache.mahout.classifier.rbm.test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.rbm.RBMClassifier;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestRBMClassifierJob extends AbstractJob {

	private static final Logger log = LoggerFactory.getLogger(TestRBMClassifierJob.class);

	public static void main(String[] args) throws Exception {
		if(args==null|| args.length==0)
			args = new String[]{
		          "--input", "/home/dirk/mnist/2chunks/chunk292",
		          "--model", "/home/dirk/mnist/30its_220chunks_20h",
		          "-lc","10"};
	    ToolRunner.run(new Configuration(), new TestRBMClassifierJob(), args);
	}
	
	/* (non-Javadoc)
	 * @see org.apache.hadoop.util.Tool#run(java.lang.String[])
	 */
	@Override
	public int run(String[] args) throws Exception {
		addInputOption();
	    addOption("model", "m", "The path to the model built during training", true);
	    addOption("labelcount", "lc", "total count of labels existent in the training set", true);
	    addOption(new DefaultOptionBuilder()
					.withLongName(DefaultOptionCreator.MAPREDUCE_METHOD)
					.withRequired(false)
					.withDescription("Run tests with map/reduce")
					.withShortName("mr").create());
	    
	    Map<String, String> parsedArgs = parseArguments(args);
	    if (parsedArgs == null) {
	      return -1;
	    }
	    
	    int labelcount = Integer.parseInt(getOption("labelcount"));

	    Path model = new Path(parsedArgs.get("--model"));
	    if(!model.getFileSystem(getConf()).exists(model)) {
	    	log.error("Model file does not exist!");
	    	return -1;
	    }
	    
	    List<String> lables= new ArrayList<String>();
	    for(int i = 0; i<labelcount; i++)
	    	lables.add(String.valueOf(i));
	    FileSystem fs = getInputPath().getFileSystem(getConf());
	    ResultAnalyzer analyzer = new ResultAnalyzer(lables, "-1");
	    Path[] batches;
	    if(fs.isFile(getInputPath()))
	    	batches = new Path[]{getInputPath()};
	    else {
	    	FileStatus[] stati = fs.listStatus(getInputPath());
	    	batches = new Path[stati.length];
	    	for (int i = 0; i < stati.length; i++) {
				batches[i] = stati[i].getPath();
			}	    		
	    }
	    
	    for (Path input : batches) {
		    if(hasOption("mapreduce")) {
			    HadoopUtil.cacheFiles(model, getConf());
			    //the output key is the expected value, the output value are the scores for all the labels
			    Job testJob = prepareJob(input, getTempPath("testresults"), SequenceFileInputFormat.class, TestRBMClassifierMapper.class,
			            				 IntWritable.class, VectorWritable.class, SequenceFileOutputFormat.class);
			    //testJob.getConfiguration().set(LABEL_KEY, parsedArgs.get("--labels"));
			    testJob.waitForCompletion(true);
			    
			    //loop over the results and create the confusion matrix
			    SequenceFileDirIterable<IntWritable, VectorWritable> dirIterable =
			        new SequenceFileDirIterable<IntWritable, VectorWritable>(getTempPath("testresults"),
					                                                          PathType.LIST,
					                                                          PathFilters.partFilter(),
					                                                          getConf());
		
			    analyzeResults(dirIterable, analyzer);
		    
		    }
		    else {
		    	analyzeResults(model, analyzer,input);
		    }	    	
	    }

	    log.info("RBMClassifier Results: {}", analyzer);
	    
	    if(executor!=null)
	    	executor.shutdownNow();
	    return 0;
	  }

	private ExecutorService executor;
	List<RBMClassifierCall> tasks;
	
	private void analyzeResults(Path model, ResultAnalyzer analyzer, Path input)
			throws IOException, InterruptedException, ExecutionException {
		int testsize =0;
		int threadCount =20;
		RBMClassifier rbmCl = RBMClassifier.materialize(model, getConf());
		if(executor==null)
    		executor = Executors.newFixedThreadPool(threadCount);
    	
    	if(tasks==null)
    		tasks = new ArrayList<RBMClassifierCall>();

		for (Pair<IntWritable, VectorWritable> record : 
			 new SequenceFileIterable<IntWritable, VectorWritable>(input,getConf())) {
			if(tasks.size()<threadCount)				
				tasks.add(new RBMClassifierCall(rbmCl.clone(), record.getSecond().get(), record.getFirst().get()));
			else {
				tasks.get(testsize%threadCount).input = record.getSecond().get();
				tasks.get(testsize%threadCount).label = record.getFirst().get();
			}
			
			if(testsize%threadCount==threadCount-1) {
				List<Future<Pair<Integer,Vector>>> futureResults = executor.invokeAll(tasks);
				for (int i = 0; i < futureResults.size(); i++) {
					  int bestIdx = Integer.MIN_VALUE;
				      double bestScore = Long.MIN_VALUE;
				      Pair<Integer, Vector> pair = futureResults.get(i).get();
				      for (Vector.Element element : pair.getSecond()) {
				        if (element.get() > bestScore) {
				          bestScore = element.get();
				          bestIdx = element.index();
				        }
				      }
				      if (bestIdx != Integer.MIN_VALUE) {
				        ClassifierResult classifierResult = new ClassifierResult(String.valueOf(bestIdx), bestScore);
				        analyzer.addInstance(String.valueOf(pair.getFirst()), classifierResult);
				      }
				}
			}
			
			testsize++;
		 }
		
		if(testsize%20!=0) {
			List<Future<Pair<Integer,Vector>>> futureResults = executor.invokeAll(tasks.subList(0, (testsize-1) %20));
			for (int i = 0; i < futureResults.size(); i++) {
				int bestIdx = Integer.MIN_VALUE;
			      double bestScore = Long.MIN_VALUE;
			      Pair<Integer, Vector> pair = futureResults.get(i).get();
			      for (Vector.Element element : pair.getSecond()) {
			        if (element.get() > bestScore) {
			          bestScore = element.get();
			          bestIdx = element.index();
			        }
			      }
			      if (bestIdx != Integer.MIN_VALUE) {
			        ClassifierResult classifierResult = new ClassifierResult(String.valueOf(bestIdx), bestScore);
			        analyzer.addInstance(String.valueOf(pair.getFirst()), classifierResult);
			      }
			}
		}
	}

	  private void analyzeResults(SequenceFileDirIterable<IntWritable, VectorWritable> dirIterable,
	                                     ResultAnalyzer analyzer) {
	    for (Pair<IntWritable, VectorWritable> pair : dirIterable) {
	      int bestIdx = Integer.MIN_VALUE;
	      double bestScore = Long.MIN_VALUE;
	      for (Vector.Element element : pair.getSecond().get()) {
	        if (element.get() > bestScore) {
	          bestScore = element.get();
	          bestIdx = element.index();
	        }
	      }
	      if (bestIdx != Integer.MIN_VALUE) {
	        ClassifierResult classifierResult = new ClassifierResult(String.valueOf(bestIdx), bestScore);
	        analyzer.addInstance(String.valueOf(pair.getFirst().get()), classifierResult);
	      }
	      
	    }
	  }
	  
	  class RBMClassifierCall implements Callable<Pair<Integer,Vector>> {
		    private RBMClassifier rbmCl;
		    private Vector input;
		    private int label;
			  
			public RBMClassifierCall(RBMClassifier rbmCl, Vector input, int label) {
				this.rbmCl = rbmCl;
				this.input = input;
				this.label = label;
			}
		  
			@Override
			public Pair<Integer,Vector> call() throws Exception {
				return new Pair<Integer, Vector>(label, rbmCl.classify(input, 10));
			}
		  
	  }
}
