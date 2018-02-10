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
  let weights: dl.Tensor1D = dl.zeros([2]);
  console.log("weights "+weights.dataSync());
  const result = dl.step(weights);
  console.log(result);
  let bias:dl.Scalar = dl.scalar(0);
  let input_vecs=dl.tensor2d([[1,1], [0,0],[1,0], [0,1]]);
  let label=dl.tensor1d([1,0,0,0]);
  console.log(label);
  console.log("input_vecs "+input_vecs.dataSync());
  console.log(bias);
  for (let i = 0; i < 10; ++i) {
    //missing zip()
  }
}

execute();