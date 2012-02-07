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
package org.apache.mahout.classifier.rbm.training;

import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;

/**
 * The Class DBMBackPropTrainingReducer.
 */
public class DBMBackPropTrainingReducer extends Reducer<IntWritable,MatrixWritable,IntWritable,MatrixWritable>{

	/* (non-Javadoc)
	 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
	 */
	protected void reduce(IntWritable key, Iterable<MatrixWritable> matrices, Context context) throws java.io.IOException ,InterruptedException {
		if(!matrices.iterator().hasNext())
			return;
		Matrix result = matrices.iterator().next().get();
		for (Iterator<MatrixWritable> iterator = matrices.iterator(); iterator.hasNext();) {
			 result = result.plus(iterator.next().get());
		}
		context.write(key, new MatrixWritable(result));
	};
}
