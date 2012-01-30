package org.apache.mahout.classifier.rbm.training;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.classifier.naivebayes.training.WeightsMapper;
import org.apache.mahout.classifier.rbm.layer.Layer;
import org.apache.mahout.classifier.rbm.layer.LogisticLayer;
import org.apache.mahout.classifier.rbm.layer.SoftmaxLayer;
import org.apache.mahout.classifier.rbm.model.LabeledSimpleRBM;
import org.apache.mahout.classifier.rbm.model.SimpleRBM;
import org.apache.mahout.classifier.rbm.network.DeepBoltzmannMachine;
import org.apache.mahout.classifier.rbm.training.RBMClassifierTrainingJob;
import org.apache.mahout.classifier.rbm.training.RBMGreedyPreTrainingMapper;
import org.apache.mahout.classifier.rbm.training.RBMGreedyPreTrainingReducer;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.easymock.ConstructorArgs;
import org.easymock.EasyMock;
import org.junit.Test;

import com.google.common.io.Closeables;

public class rbmClassifierTest extends MahoutTestCase {
	
	private Path output;
	private Path input;
	private Configuration conf;
	private FileSystem fileSystem;
	private DeepBoltzmannMachine dbm;
	private Pair<Integer,Vector> testPair;

	@Override
	public void setUp() throws Exception {
		input = getTestTempFilePath("input");
		output = getTestTempDirPath("output");
		conf = new Configuration();
		
		fileSystem = input.getFileSystem(conf);
		fileSystem.delete(output, true);
		fileSystem.mkdirs(input.getParent());
		fileSystem.create(input, true);
		fileSystem.mkdirs(output);
		
		Writer writer = SequenceFile.createWriter(fileSystem, conf, input, IntWritable.class, VectorWritable.class);
		
		//TODO randomutils gave always the same value??
		Random rand = new Random();

		for (int i = 0; i < 4; i++) {
			VectorWritable vectorWritable = new VectorWritable(
					  new DenseVector(new double[]{rand.nextDouble(),
										  rand.nextDouble(),
										  rand.nextDouble(),
										  rand.nextDouble(),
										  rand.nextDouble()}));
			if(i==0)
				testPair = new Pair<Integer, Vector>(0, vectorWritable.get());
			writer.append(new IntWritable(i%2),
						  vectorWritable);
		}
		
		Closeables.closeQuietly(writer);
		
		String[] layers = {"5","10","10"};
		Layer[] layers2 = new Layer[layers.length];
				
		layers2[0]= new LogisticLayer(Integer.parseInt(layers[0]));
		for (int i = 1; i < layers.length; i++) {
			layers2[i]= new LogisticLayer(Integer.parseInt(layers[i]));
			if(i==1)
				dbm = new DeepBoltzmannMachine(new SimpleRBM(layers2[0], layers2[1]));
			else if(i==layers.length-1)
				dbm.stackRBM(new LabeledSimpleRBM(layers2[i-1],layers2[i],new SoftmaxLayer(2)));
			else
				dbm.stackRBM(new SimpleRBM(layers2[i-1],layers2[i]));
	      }
		
		dbm.serialize(output, conf);
		
		super.setUp();
	}
	
	/*@Test
	public void testMapper() throws IOException, InterruptedException, NoSuchFieldException, IllegalAccessException {
	    Mapper.Context ctx = EasyMock.createMock(Mapper.Context.class);
		 
		    Vector instance1 = new DenseVector(new double[] { 1, 0,   0.5, 0.5, 0 });

		    RBMClassifierTrainingMapper mapper = new RBMClassifierTrainingMapper();
		    setField(mapper, "dbm", dbm);
		    
		    mapper.map(new VectorWritable(instance1), new VectorWritable(instance1), ctx);
	}*/
	
	@Test
 	public void testClassification() throws Exception {				
		
		/*DeepBoltzmannMachine dbm = null;
		String[] layers = {"10","15","20"};
		Layer layer2 = new LogisticLayer(Integer.parseInt(layers[0]));

		for (int i = 0; i < layers.length-1; i++) {
			Layer layer = layer2;
			layer2 = new LogisticLayer(Integer.parseInt(layers[i+1]));
			if(i==0)
				dbm = new DeepBoltzmannMachine(new SimpleRBM(layer, layer2));
			else if(i==layers.length-2)
				dbm.stackRBM(new LabeledSimpleRBM(layer,layer2,new SoftmaxLayer(10)));
			else
				dbm.stackRBM(new SimpleRBM(layer, layer2));
	      }
	      
	      dbm.serialize(output, conf);*/
		
		
		RBMClassifierTrainingJob job = new RBMClassifierTrainingJob();
		job.setConf(conf);
		String[] args = {
		          optKey(DefaultOptionCreator.INPUT_OPTION), input.toUri().getPath(),
		          optKey(DefaultOptionCreator.OUTPUT_OPTION), output.toUri().getPath(),
		         //optKey("structure"), "5,5,5",
		          optKey("labelcount"), "2"	,
		          optKey("maxIter"), "2"};

		double errorBefore = dbm.getRBM(0).getReconstructionError(testPair.getSecond());
		assertEquals(0, job.run(args));
		DeepBoltzmannMachine dbm2 = DeepBoltzmannMachine.materialize(output, conf);
		double errorAfter = dbm2.getRBM(0).getReconstructionError(testPair.getSecond());

		assertTrue(errorAfter<=errorBefore);
	}
	
	@Override
	public void tearDown() throws Exception {
		fileSystem.delete(input.getParent(), true);
		super.tearDown();
	}
}
