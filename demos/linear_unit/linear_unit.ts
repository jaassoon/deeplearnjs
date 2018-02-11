/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as dl from 'deeplearn';

export async function execute(event?: Event) {
  let weights = dl.zeros([1]) as dl.Tensor1D;
  let bias = dl.scalar(.0) as dl.Scalar;
  let input_vecs=[[5], [3], [8], [1.4], [10.1]];
  let label=[5500, 2300, 7600, 1800, 11400];
  let rate=0.01;
  for (let i = 0; i < 10; ++i) {
    for (let j = 0; j < input_vecs.length; ++j) {
      let input_vec=dl.tensor1d(input_vecs[j]);
      let output=dl.sum(input_vec.mul(weights)).add(bias);
      let delta=dl.scalar(label[j]).sub(output);
      weights=weights.add(input_vec.mul(delta).mul(dl.scalar(rate)));
      bias=bias.add(delta.mul(dl.scalar(rate)));
    }
  }
  console.log('weights='+weights.dataSync())
  console.log('bias='+bias.dataSync())
}

execute();