package org.apache.mahout.classifier.rbm.test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.naivebayes.BayesUtils;
import org.apache.mahout.classifier.naivebayes.test.BayesTestMapper;
import org.apache.mahout.classifier.naivebayes.test.TestNaiveBayesDriver;
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

	public static final String LABEL_KEY = "labels";
	public static final String COMPLEMENTARY = "class"; //b for bayes, c for complementary
	
	public static void main(String[] args) throws Exception {
	    ToolRunner.run(new Configuration(), new TestRBMClassifierJob(), args);
	}
	
	@Override
	public int run(String[] args) throws Exception {
		addInputOption();
	    addOutputOption();
	    addOption(addOption(DefaultOptionCreator.overwriteOption().create()));
	    addOption("model", "m", "The path to the model built during training", true);
	    addOption("labelcount", "lc", "total count of labels existent in the training set", true);

	    Map<String, String> parsedArgs = parseArguments(args);
	    if (parsedArgs == null) {
	      return -1;
	    }
	    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
	      HadoopUtil.delete(getConf(), getOutputPath());
	    }
	    
	    int labelcount = Integer.parseInt(getOption("labelcount"));

	    Path model = new Path(parsedArgs.get("--model"));
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
	    List<String> lables= new ArrayList<String>();
	    for(int i = 0; i<labelcount; i++)
	    	lables.add(String.valueOf(i));
	    
	    ResultAnalyzer analyzer = new ResultAnalyzer(lables, "0");
	    analyzeResults(dirIterable, analyzer);

	    log.info("RBMClassifier Results: {}", analyzer);
	    return 0;
	  }

	  private static void analyzeResults(SequenceFileDirIterable<IntWritable, VectorWritable> dirIterable,
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
	        analyzer.addInstance(pair.getFirst().toString(), classifierResult);
	      }
	      
	    }
	  }
	  
}
