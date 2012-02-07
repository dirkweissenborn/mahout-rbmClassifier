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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.rbm.RBMClassifier;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * The Class TestRBMClassifierMapper which does the actual classifying.
 */
public class TestRBMClassifierMapper extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable>{

	/** The rbm classifier. */
	private RBMClassifier rbmCl;
	/** The rbm classifier. */
	private int iterations;

	/* (non-Javadoc)
	 * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
	 */
	protected void setup(Context context) throws java.io.IOException ,InterruptedException {
		Configuration conf = context.getConfiguration();
		Path p = HadoopUtil.cachedFile(conf);
		rbmCl = RBMClassifier.materialize(p, conf);
	    iterations = Integer.parseInt(context.getConfiguration().get("maxIter"));
	};
	
	/* (non-Javadoc)
	 * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
	 */
	protected void map(IntWritable key, VectorWritable value, Context context) throws java.io.IOException ,InterruptedException {
		Vector result;
		synchronized(rbmCl) {
			result = rbmCl.classify(value.get(),iterations);
		}
		context.write(key, new VectorWritable(result));		
	};
}
