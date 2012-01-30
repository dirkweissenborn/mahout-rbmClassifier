package org.apache.mahout.classifier.rbm.test;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VectorWritable;

public class TestRBMClassifierMapper extends Mapper<Text, VectorWritable, Text, VectorWritable>{

}
