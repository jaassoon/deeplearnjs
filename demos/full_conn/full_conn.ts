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

class FullConnectedLayer{
  input_size:number;
  output_size:number;
  W:dl.Tensor2D;
  b:dl.Tensor2D;
  input:dl.Tensor2D;
  delta:dl.Tensor2D;
  W_grad:dl.Tensor2D;
  b_grad:dl.Tensor2D;
  output:dl.Tensor2D;
  constructor(input_size:number,output_size:number){
    this.input_size = input_size
    this.output_size = output_size
    this.W = dl.randomUniform([output_size, input_size],-0.1, 0.1)
    this.b = dl.zeros([output_size,1])
    this.output = dl.zeros([output_size,1])
  }
  forward(input_array:dl.Tensor2D){
    this.input = input_array
    this.output=dl.sigmoid(this.W.matMul(input_array).add(this.b))
  }
  backward(delta_array:dl.Tensor2D){
    this.delta=sigmoid_backward(this.input).mul(this.W.transpose().matMul(delta_array))
    this.W_grad=delta_array.mul(this.input.transpose())
    this.b_grad = delta_array
  }
  update(learning_rate:number){
    this.W =this.W.add(dl.scalar(learning_rate).mul(this.W_grad))
    this.b =this.b.add(dl.scalar(learning_rate).mul(this.b_grad))
  }
  dump(){
    console.log(this.W.dataSync(),this.b.dataSync())
  }
}
export class Network{
  layers:FullConnectedLayer[]
  constructor(layers:number[]){
    this.layers = []
    for(let i=0;i<layers.length-1;i++){
      this.layers.push(new FullConnectedLayer(layers[i], layers[i+1]))
    }
  }
  predict(sample:dl.Tensor2D){
    let output = sample
    for(let i=0;i<this.layers.length;i++){
      let fcLayer=this.layers[i];
      fcLayer.forward(output)
      output=fcLayer.output
    }
    return output
  }
  train(labels:dl.Tensor2D[], data_set:dl.Tensor2D[], rate:number, mini_batch:number){
    mini_batch=10;//FIXME
    for(let i=0;i<mini_batch;i++){
      for(let d=0;d<data_set.length;d++){
        this.train_one_sample(labels[d],data_set[d],rate)
        if(d==0||d==255){
          console.log('mini_batch:%d,d:%d',i,d)
          // this.dump()
        }
      }
    }
  }
  train_one_sample(label:dl.Tensor2D, sample:dl.Tensor2D, rate:number){
    this.predict(sample)
    this.calc_gradient(label)
    this.update_weight(rate)
  }
  calc_gradient(label:dl.Tensor2D){
    let layerLast=this.layers[this.layers.length-1],
    tmpOutput=layerLast.output.mul(label.sub(layerLast.output)),
    delta=tmpOutput.mul(dl.scalar(1).sub(tmpOutput)) as dl.Tensor2D
    for(let i=this.layers.length-1;i>-1;i--){
      this.layers[i].backward(delta)
      delta=this.layers[i].delta
    }
    return delta
  }
  update_weight(rate:number){
    for(let i=0;i<this.layers.length;i++){
      this.layers[i].update(rate)
    }
  }
  dump(){
    for(let i=0;i<this.layers.length;i++){
      this.layers[i].dump()
    }
  }
  loss(output:dl.Tensor2D, label:dl.Tensor2D){
    return label.sub(output).square().mul(dl.scalar(.5)).sum().dataSync()[0]
  }
  gradient_check(sample_feature:any, sample_label:any){
    this.predict(sample_feature)
    this.calc_gradient(sample_label)
    let epsilon = 10e-4
    for(let k=0;k<this.layers.length;k++){
      let fc=this.layers[k];
      for(let i=0;i<fc.W.shape[0];i++){
        for(let j=0;j<fc.W.shape[1];j++){
          let tmpValue=fc.W.get(fc.W.locToIndex([i,j])),
          buffer=fc.W.buffer();
          buffer.set(tmpValue+epsilon,i,j)
          fc.W=buffer.toTensor()

          let output = this.predict(sample_feature),
          err1 = this.loss(sample_label, output)
          tmpValue=fc.W.get(fc.W.locToIndex([i,j]))
          buffer=fc.W.buffer();
          buffer.set(tmpValue-2*epsilon,i,j)
          fc.W=buffer.toTensor()

          output = this.predict(sample_feature)
          let err2 = this.loss(sample_label, output),
          expect_grad = (err1 - err2) / (2 * epsilon)

          tmpValue=fc.W.get(fc.W.locToIndex([i,j]))
          buffer=fc.W.buffer();
          buffer.set(tmpValue+epsilon,i,j)
          fc.W=buffer.toTensor()
          console.log('weights(%d,%d): expected - actural %.4e - %.4e',
            i,j,expect_grad, fc.W_grad.get(fc.W_grad.locToIndex([i,j])))
        }
      }
    }
  }
}
class Normalizer{
  private mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]
  norm(number:number){
    let ret=[];
    for(let i=0;i<this.mask.length;i++){
      if(number & this.mask[i])ret.push(.9)
      else ret.push(.1)
    }
    return dl.reshape(dl.tensor1d(ret),[8,1]) as dl.Tensor2D
  }
  denorm(vec:dl.Tensor2D){
    let binary=[],
    vecData=vec.dataSync()
    for(let i=0;i<vecData.length;i++){
      if(vecData[i]>0.5)binary.push(1)
      else binary.push(0)
    }
    for(let i=0;i<this.mask.length;i++){
      binary[i] = binary[i] * this.mask[i]
    }
    let ret=.0
    for(let i=0;i<binary.length;i++){
      ret+=binary[i]
    }
    return ret
  }
}
function train_data_set(){
  let normalizer=new Normalizer(),
  data_set = [] as dl.Tensor2D[],
  labels = [] as dl.Tensor2D[],
  ret=[] as any[]
  for(let i=0;i<256;i++){
    let n = normalizer.norm(Number(dl.util.randUniform(0, 256).toFixed(0)))
    data_set.push(n)
    labels.push(n)
  }
  ret.push(labels)
  ret.push(data_set)
  return ret;
  }
function correct_ratio(network:Network){
  let normalizer = new Normalizer()
  let correct = 0.0;
  for(let i=0;i<256;i++){
    if(normalizer.denorm(network.predict(normalizer.norm(i)))==i)
      correct+=1.0
  }
  console.log('correct_ratio: %f%',correct/256*100)
}
function test(){
  // labels, data_set = transpose(train_data_set())
  let dataSet= train_data_set(),
  labels=dataSet[0] as dl.Tensor2D[],
  data_set = dataSet[1] as dl.Tensor2D[],
  net = new Network([8, 3, 8]),
  rate = 0.5,
  mini_batch = 20,
  epoch = 10;
  epoch=1;//FIXME
  for(let i=0;i<epoch;i++){
    net.train(labels, data_set, rate, mini_batch)
    console.log('after epoch %d loss: %f',(i + 1)
      ,net.loss(labels[labels.length-1], net.predict(data_set[data_set.length-1]))
    )
    rate /= 2
  }
  correct_ratio(net)
}
function sigmoid_backward(
  output:dl.Tensor2D){
  return output.mul(dl.scalar(1).sub(output))
}
function gradient_check(){
  // labels, data_set = transpose(train_data_set())
  let dataSet= train_data_set(),
  labels=dataSet[0] as dl.Tensor2D[],
  data_set = dataSet[1] as dl.Tensor2D[],
  net = new Network([8, 3, 8])
  net.gradient_check(data_set[0], labels[0])
  return net
}
// test()
// gradient_check()