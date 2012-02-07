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
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The Class TestRBMClassifierJob which runs the tests in map/reduce or locally multithreaded.
 */
public class TestRBMClassifierJob extends AbstractJob {

	/** The Constant log. */
	private static final Logger log = LoggerFactory.getLogger(TestRBMClassifierJob.class);

	/**
	 * The main method.
	 *
	 * @param args the arguments
	 * @throws Exception the exception
	 */
	public static void main(String[] args) throws Exception {
	    ToolRunner.run(new Configuration(), new TestRBMClassifierJob(), args);
	}

	private int iterations;
	
	/* (non-Javadoc)
	 * @see org.apache.hadoop.util.Tool#run(java.lang.String[])
	 */
	@Override
	public int run(String[] args) throws Exception {
		addInputOption();
	    addOption("model", "m", "The path to the model built during training", true);
	    addOption("labelcount", "lc", "total count of labels existent in the training set", true);
	    addOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION, "max", 
	    		"least number of stable iterations in classification layer when classifying","10");
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
	    iterations = Integer.parseInt(getOption("maxIter"));
	    
	    //check models existence
	    Path model = new Path(parsedArgs.get("--model"));
	    if(!model.getFileSystem(getConf()).exists(model)) {
	    	log.error("Model file does not exist!");
	    	return -1;
	    }
	    
	    //create the list of all labels
	    List<String> lables= new ArrayList<String>();
	    for(int i = 0; i<labelcount; i++)
	    	lables.add(String.valueOf(i));
	    
	    FileSystem fs = getInputPath().getFileSystem(getConf());
	    ResultAnalyzer analyzer = new ResultAnalyzer(lables, "-1");
	    //initiate the paths to the test batches
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
	    
	    if(hasOption("mapreduce"))
	    	HadoopUtil.delete(getConf(), getTempPath("testresults"));
	    
	    for (Path input : batches) {
		    if(hasOption("mapreduce")) {
			    HadoopUtil.cacheFiles(model, getConf());
			    //the output key is the expected value, the output value are the scores for all the labels
			    Job testJob = prepareJob(input, getTempPath("testresults"), SequenceFileInputFormat.class, TestRBMClassifierMapper.class,
			            				 IntWritable.class, VectorWritable.class, SequenceFileOutputFormat.class);
			    testJob.getConfiguration().set("maxIter", String.valueOf(iterations));
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
		    	//test job locally
		    	runTestsLocally(model, analyzer,input);
		    }	    	
	    }

	    //output the result of the tests
	    log.info("RBMClassifier Results: {}", analyzer);
	    
	    //stop all running threads
	    if(executor!=null)
	    	executor.shutdownNow();
	    return 0;
	  }

	/** The executor. */
	private ExecutorService executor;
	
	/** The tasks. */
	List<RBMClassifierCall> tasks;
	
	/**
	 * Analyze results locally.
	 *
	 * @param model the model
	 * @param analyzer the analyzer
	 * @param input the input
	 * @throws IOException Signals that an I/O exception has occurred.
	 * @throws InterruptedException the interrupted exception
	 * @throws ExecutionException the execution exception
	 */
	private void runTestsLocally(Path model, ResultAnalyzer analyzer, Path input)
			throws IOException, InterruptedException, ExecutionException {
		int testsize =0;
		//maximum number of threads that are used, I think 20 is ok
		int threadCount =20;
		RBMClassifier rbmCl = RBMClassifier.materialize(model, getConf());
		//initialize the executor if not already done
		if(executor==null)
    		executor = Executors.newFixedThreadPool(threadCount);
    	//initialize the tasks, which are run by the executor
    	if(tasks==null)
    		tasks = new ArrayList<RBMClassifierCall>();

		for (Pair<IntWritable, VectorWritable> record : 
			 new SequenceFileIterable<IntWritable, VectorWritable>(input,getConf())) {
			//prepare the tasks
			if(tasks.size()<threadCount)				
				tasks.add(new RBMClassifierCall(rbmCl.clone(), record.getSecond().get(), record.getFirst().get(), iterations));
			else {
				tasks.get(testsize%threadCount).input = record.getSecond().get();
				tasks.get(testsize%threadCount).label = record.getFirst().get();
			}
			
			//run the tasks
			if(testsize%threadCount==threadCount-1) {
				List<Future<Pair<Integer,Vector>>> futureResults = executor.invokeAll(tasks);
				//analyze results
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
		
		//run and analyze remaining tasks
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

	 /**
  	 * Analyze results of M/R job.
  	 *
  	 * @param dirIterable the directory with the results
  	 * @param analyzer the analyzer
  	 */
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
	  
	  /**
  	 * The Class RBMClassifier is the callable thread for the local classifying task.
  	 */
  	class RBMClassifierCall implements Callable<Pair<Integer,Vector>> {
		    
    		/** The rbm cl. */
    		private RBMClassifier rbmCl;
		    
    		/** The input. */
    		private Vector input;
		    
    		/** The label. */
    		private int label;

    		/** The iterations. */
			private int iterations;
			  
			/**
			 * Instantiates a new rBM classifier call.
			 *
			 * @param rbmCl the rbm cl
			 * @param input the input
			 * @param label the label
			 * @param iterations the number of iterations until stable
			 */
			public RBMClassifierCall(RBMClassifier rbmCl, Vector input, int label, int iterations) {
				this.rbmCl = rbmCl;
				this.input = input;
				this.label = label;
				this.iterations = iterations;
			}
		  
			/* (non-Javadoc)
			 * @see java.util.concurrent.Callable#call()
			 */
			@Override
			public Pair<Integer,Vector> call() throws Exception {
				return new Pair<Integer, Vector>(label, rbmCl.classify(input, iterations));
			}
		  
	  }
}
