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
  let weights = dl.zeros([2]) as dl.Tensor1D;
  let bias = dl.scalar(.0) as dl.Scalar;
  // let input_vecs=dl.tensor2d([[1,1], [0,0],[1,0], [0,1]]);
  // let label=dl.tensor1d([1,0,0,0]);
  for (let i = 0; i < 10; ++i) {
    for (let j = 0; j < 4; ++j) {
      let input_vec=dl.tensor1d([1,1]);
      if(j==1){
        input_vec=dl.tensor1d([0,0]);
      }else if(j==2){
        input_vec=dl.tensor1d([1,0]);
      }else if(j==3){
        input_vec=dl.tensor1d([0,1]);
      }
      let output=dl.step(dl.sum(input_vec.mul(weights)).add(bias));
      let delta=dl.scalar(.0);
      if(j==0)delta=dl.scalar(1.0).sub(output);
      else delta=dl.scalar(.0).sub(output);
      weights=weights.add(input_vec.mul(delta).mul(dl.scalar(0.1)));
      console.log('weights'+j+' '+weights.dataSync());
      bias=bias.add(delta.mul(dl.scalar(0.1)));
      console.log('bias'+j+' '+bias.dataSync());
    }
  }
  console.log(weights.dataSync())
  console.log(bias.dataSync())
}

execute();