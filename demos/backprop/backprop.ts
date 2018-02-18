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
class Node{
  layer_index:number;
  node_index:number;
  downstream:Connection[];
  upstream:Connection[];
  output=dl.scalar(0) as dl.Scalar;
  delta=dl.scalar(0) as dl.Scalar;
  constructor(layer_index:number, node_index:number){
    this.layer_index = layer_index;
    this.node_index = node_index;
    this.upstream=new Array<Connection>();
    this.downstream=new Array<Connection>();
  }
  set_output(output:number){this.output=dl.scalar(output)}
  append_downstream_connection(conn:Connection){this.downstream.push(conn)}
  append_upstream_connection(conn:Connection){this.upstream.push(conn)}
  calc_output(){
    let output=dl.scalar(0);
    for(let i=0;i<this.upstream.length;i++){
      let conn=this.upstream[i];
      output=conn.upstream_node.output.mul(conn.weight).add(output);
    }
    this.output = dl.sigmoid(output);
  }
  calc_hidden_layer_delta(){
    let downstream_delta=dl.scalar(.0);
    for(let i=0;i<this.downstream.length;i++){
      let conn=this.downstream[i];
      downstream_delta=conn.downstream_node.delta.mul(conn.weight).add(downstream_delta);
    }
    this.delta = this.output.mul(dl.scalar(1).sub(this.output)).mul(downstream_delta)
  }
  calc_output_layer_delta(label:number){
    this.delta=this.output.mul(dl.scalar(1).sub(this.output)).mul(dl.scalar(label).sub(this.output))
  }
  toString(){
    let node_str = '%d-%d: output: %f delta: %f';
    // let downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), this.downstream, '')
    // let upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), this.upstream, '')
    // return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str 
    console.log(node_str,this.layer_index,this.node_index,
      Number(this.output.dataSync()[0].toFixed(6)),
      Number(this.delta.dataSync()[0].toFixed(6)))
  }
}
class ConstNode{
  layer_index:number;
  node_index:number;
  output=dl.scalar(1);
  delta=dl.scalar(0);
  downstream:Connection[];
  constructor(layer_index:number,node_index:number){
    this.layer_index = layer_index;
    this.node_index = node_index;
    this.downstream = [];
  }
  append_downstream_connection(conn:Connection){this.downstream.push(conn)}
  calc_hidden_layer_delta(){
    let downstream_delta=dl.scalar(.0);
    for(let i=0;i<this.downstream.length;i++){
      let conn=this.downstream[i];
      downstream_delta=conn.downstream_node.delta.mul(conn.weight).add(downstream_delta);
    }
    this.delta = this.output.mul(dl.scalar(1).sub(this.output)).mul(downstream_delta)
  }
  toString(){
    let node_str = '${this.layer_index}-${this.node_index}: output: 1';
    // downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
    // return node_str + '\n\tdownstream:' + downstream_str
    console.log(node_str)
  }
}
class Connection{
  upstream_node:Node;
  downstream_node:Node;
  weight:dl.Scalar;
  protected gradient=dl.scalar(.0) as dl.Scalar;
  constructor(upstream_node:Node, downstream_node:Node){
    this.upstream_node = upstream_node
    this.downstream_node = downstream_node;
    this.weight =dl.scalar(dl.util.randUniform(-0.1, 0.1));
  }
  calc_gradient(){
    this.gradient = this.downstream_node.delta.mul(this.upstream_node.output)
  }
  update_weight(rate:number){
    this.calc_gradient()
    this.weight = this.gradient.mul(dl.scalar(rate)).add(this.weight)
  }
  get_gradient(){return this.gradient}
  toString(){
    console.log(`(
      ${this.upstream_node.layer_index}-
      ${this.upstream_node.node_index})->(
      ${this.downstream_node.layer_index}-
      ${this.downstream_node.node_index})=
      ${this.weight}`);
  }
}
class Layer{
  protected layer_index:number;
  nodes:any[];
  constructor(layer_index:number, node_count:number){
    this.layer_index = layer_index
    this.nodes = []
    for(let i=0;i<node_count;i++){
      this.nodes.push(new Node(layer_index, i))
    }
    this.nodes.push(new ConstNode(layer_index, node_count))
  }
  set_output(data:number[]){
    for(let i=0;i<data.length;i++){
      this.nodes[i].set_output(data[i])
    }
  }
  calc_output(){
    for(let i=0;i<this.nodes.length-1;i++)
      this.nodes[i].calc_output()
  }
  dump(){
    for(let i=0;i<this.nodes.length;i++)
      this.nodes[i].toString()
  }
}
class Network{
  protected connections:Connections;
  protected layers:Layer[];
  constructor(layers:any[]) {
    this.connections = new Connections();
    this.layers=new Array<Layer>();
    const layer_count = layers.length;//[8,3,8]
    for(let i=0;i<layer_count;i++){this.layers.push(new Layer(i,layers[i]))}
    for(let i=0;i<layer_count-1;i++){//layer_count=3
      let connections=[];
      for(let j=0;j<this.layers[i].nodes.length;j++){
        let upstream_node=this.layers[i].nodes[j];
        for (let k = 0; k < this.layers[i+1].nodes.length-1; k++) {
          let downstream=this.layers[i+1].nodes[k];
          connections.push(new Connection(upstream_node,downstream))
        }
      }
      for(let i=0;i<connections.length;i++){
        let conn=connections[i];
        conn.downstream_node.append_upstream_connection(conn)
        conn.upstream_node.append_downstream_connection(conn)
        this.connections.add_connection(conn)
      }
    }
  }
  train(labels:any[], data_set:any[], rate:number, epoch:number){
    for(let i=0;i<epoch;i++){
      if(i==1)break;//FIXME
      for(let d=0;d<data_set.length;d++){
        if(d==3)break;//FIXME
        this.train_one_sample(labels[d], data_set[d], rate)
        console.log('sample %d training finished',d)
      }
    }
  }
  train_one_sample(label:number[], sample:number[], rate:number){
    this.predict(sample)
    this.calc_delta(label)
    this.update_weight(rate)
  }
  calc_delta(label:any[]){
    let layersCnt=this.layers.length;
    let output_nodes = this.layers[layersCnt-1].nodes
    for(let i=0;i<label.length;i++){
      output_nodes[i].calc_output_layer_delta(label[i])
    }
    for(let i=layersCnt-2,layer=this.layers[i];i<layersCnt;i++){
      for(let j=0,node=layer.nodes[j];j<layer.nodes.length;j++){
        node.calc_hidden_layer_delta()
      }
    }
  }
  update_weight(rate:number){
    for(let i=0;i<this.layers.length-1;i++){
      let layer=this.layers[i]
      for(let j=0;j<layer.nodes.length;j++){
        let node=layer.nodes[j]
        for(let k=0;k<node.downstream.length;k++){
          node.downstream[k].update_weight(rate)
        }
      }
    }
  }
  calc_gradient(){
    for(let i=0;i<this.layers.length-1;i++){
      let layer=this.layers[i]
      for(let j=0;j<layer.nodes.length;j++){
        let node=layer.nodes[j]
        for(let k=0;k<node.downstream.length;k++){
          node.downstream[k].calc_gradient()
        }
      }
    }
  }
  get_gradient(label:any[], sample:any[]){
    this.predict(sample)
    this.calc_delta(label)
    this.calc_gradient()
  }
  predict(sample:number[]){
    this.layers[0].set_output(sample)
    for(let i=1;i<this.layers.length;i++){
      this.layers[i].calc_output()
    }
    let layer=this.layers[this.layers.length-1]
    let ret=[];
    for(let i=0;i<layer.nodes.length-1;i++){
      ret.push(layer.nodes[i].output)
    }
    return ret;
  }
  dump(){
    for(let i=0;i<this.layers.length;i++){
      this.layers[i].dump()
    }
  }
}
class Connections{
  protected conn:Connection[];
  constructor(){this.conn=new Array<Connection>()}
  add_connection(connection:Connection){this.conn.push(connection);}
  dump(){for(let i=0;i<this.conn.length;i++)this.conn[i].toString()}
}
class Normalizer{
  private mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]
  norm(number:number){
    let ret=new Array<number>();
    for(let i=0;i<this.mask.length;i++){
      if(number & this.mask[i])ret.push(.9)
      else ret.push(.1)
    }
    return ret;
  }
   // denorm( vec):
   //     binary = map(lambda i: 1 if i > 0.5 else 0, vec)
   //     for i in range(len(self.mask)):
   //         binary[i] = binary[i] * self.mask[i]
   //      return reduce(lambda x,y: x + y, binary)
}
function train_data_set(){
  let normalizer=new Normalizer();
  let data_set = [] as any[]
  let labels = [] as any[]
  let ret=[] as any[]
  for(let i=0;i<256;i=i+8){
    let n = normalizer.norm(Number(dl.util.randUniform(0, 256).toFixed(0)))
    data_set.push(n)
    labels.push(n)
  }
  ret.push(labels)
  ret.push(data_set)
  return ret;
  }
function train(network:Network){
  let dataSet= train_data_set()
  network.train(dataSet[0], dataSet[1], 0.3, 50)
}
const net=new Network([8,3,8]);
console.log(net)
train(net)
net.dump()