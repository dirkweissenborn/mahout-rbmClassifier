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
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestRBMClassifierJob extends AbstractJob {

	private static final Logger log = LoggerFactory.getLogger(TestRBMClassifierJob.class);

	public static void main(String[] args) throws Exception {
	    ToolRunner.run(new Configuration(), new TestRBMClassifierJob(), args);
	}
	
	@Override
	public int run(String[] args) throws Exception {
		addInputOption();
	    addOutputOption();
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
	    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
	      HadoopUtil.delete(getConf(), getOutputPath());
	    }
	    
	    int labelcount = Integer.parseInt(getOption("labelcount"));

	    Path model = new Path(parsedArgs.get("--model"));
	    
	    List<String> lables= new ArrayList<String>();
	    for(int i = 0; i<labelcount; i++)
	    	lables.add(String.valueOf(i));
	    
	    ResultAnalyzer analyzer = new ResultAnalyzer(lables, "0");
	    
	    if(hasOption("mapreduce")) {
		    HadoopUtil.cacheFiles(model, getConf());
		    //the output key is the expected value, the output value are the scores for all the labels
		    Job testJob = prepareJob(getInputPath(), getOutputPath(), SequenceFileInputFormat.class, TestRBMClassifierMapper.class,
		            				 IntWritable.class, VectorWritable.class, SequenceFileOutputFormat.class);
		    //testJob.getConfiguration().set(LABEL_KEY, parsedArgs.get("--labels"));
		    testJob.waitForCompletion(true);
		    
		    //loop over the results and create the confusion matrix
		    SequenceFileDirIterable<IntWritable, VectorWritable> dirIterable =
		        new SequenceFileDirIterable<IntWritable, VectorWritable>(getOutputPath(),
				                                                          PathType.LIST,
				                                                          PathFilters.partFilter(),
				                                                          getConf());
	
		    analyzeResults(dirIterable, analyzer);
	    
	    }
	    else {
	    	analyzeResults(model, analyzer);
	    }	    	

	    log.info("RBMClassifier Results: {}", analyzer);
	    return 0;
	  }

	private ExecutorService executor;
	List<RBMClassifierCall> tasks;
	
	private void analyzeResults(Path model, ResultAnalyzer analyzer)
			throws IOException, InterruptedException, ExecutionException {
		int testsize =0;
		int threadCount =20;
		RBMClassifier rbmCl = RBMClassifier.materialize(model, getConf());
		if(executor==null)
    		executor = Executors.newFixedThreadPool(threadCount);
    	
    	if(tasks==null)
    		tasks = new ArrayList<RBMClassifierCall>();

		for (Pair<IntWritable, VectorWritable> record : 
			 new SequenceFileIterable<IntWritable, VectorWritable>(getInputPath(),getConf())) {
			if(tasks.size()<threadCount)				
				tasks.add(new RBMClassifierCall(rbmCl.clone(), record.getSecond().get()));
			else {
				tasks.get(testsize%threadCount).input = record.getSecond().get();
			}
			
			if(testsize%threadCount==threadCount-1) {
				List<Future<Vector>> futureResults = executor.invokeAll(tasks);
				for (int i = 0; i < futureResults.size(); i++) {
					int bestIdx = Integer.MIN_VALUE;
				      double bestScore = Long.MIN_VALUE;
				      for (Vector.Element element : futureResults.get(i).get()) {
				        if (element.get() > bestScore) {
				          bestScore = element.get();
				          bestIdx = element.index();
				        }
				      }
				      if (bestIdx != Integer.MIN_VALUE) {
				        ClassifierResult classifierResult = new ClassifierResult(String.valueOf(bestIdx), bestScore);
				        analyzer.addInstance(String.valueOf(record.getFirst().get()), classifierResult);
				      }
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
	  
	  class RBMClassifierCall implements Callable<Vector> {
		    private RBMClassifier rbmCl;
		    private Vector input;
			  
			public RBMClassifierCall(RBMClassifier rbmCl, Vector input) {
				this.rbmCl = rbmCl;
				this.input = input;
			}
		  
			@Override
			public Vector call() throws Exception {
				return rbmCl.classify(input, 10);
			}
		  
	  }
}
