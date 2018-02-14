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
  protected layer_index:number;
  protected node_index:number;
  protected downstream:Connection[];
  protected upstream:Connection[];
  protected output=0;
  protected delta=0;
  constructor(layer_index:number, node_index:number){
    this.layer_index = layer_index;
    this.node_index = node_index;
  }
  set_output(output:number){this.output=output}
  append_downstream_connection(conn:Connection){this.downstream.push(conn)}
  append_upstream_connection(conn:Connection){this.upstream.push(conn)}
  calc_output(){
    // let output = dl.reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, this.upstream, 0)
    // this.output = dl.sigmoid(output);
  }
  calc_hidden_layer_delta(){
        // downstream_delta = reduce(
        //     lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
        //     this.downstream, 0.0)
        // this.delta = this.output * (1 - this.output) * downstream_delta
  }
  calc_output_layer_delta(label:number){this.delta = this.output * (1 - this.output) * (label - this.output)}
    // __str__(){
    //     node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
    //     downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
    //     upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
    //     return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str 
}
class ConstNode{
  protected layer_index:number;
  protected node_index:number;
  protected output=1;
  protected downstream:Connection[];
  constructor(layer_index:number,node_index:number){
    this.layer_index = layer_index;
    this.node_index = node_index;
    this.downstream = [];
  }
  append_downstream_connection(conn:Connection){
    this.downstream.push(conn);
  }
  calc_hidden_layer_delta(){
        // downstream_delta = reduce(
        //     lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
        //     this.downstream, 0.0)
        // this.delta = this.output * (1 - this.output) * downstream_delta
  }
     /*__str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str*/
}
class Connection{
  upstream_node:Node;
  downstream_node:Node;
  protected weight:number;
  protected gradient:number;
  constructor(upstream_node:Node, downstream_node:Node){
    this.upstream_node = upstream_node
    this.downstream_node = downstream_node
    // this.weight = dl.uniform(-0.1, 0.1)
    this.gradient = 0.0
  }
  calc_gradient(){
    // this.gradient = this.downstream_node.delta * this.upstream_node.output
  }
  update_weight(rate:number){
    this.calc_gradient()
    this.weight += rate * this.gradient
  }
  get_gradient(){return this.gradient}

  toString(){
    console.log(`
    // return '(%u-%u) -> (%u-%u) = %f' % (
      $this.upstream_node.layer_index, 
      $this.upstream_node.node_index,
      $this.downstream_node.layer_index, 
      $this.downstream_node.node_index, 
      $this.weight
      `)
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
  set_output(data:any[]){
    for(let i=0;i<data.length;i++){
      this.nodes[i].set_output(data[i])
    }
  }
  calc_output(){
    for(let i=0,node=this.nodes[i];i<this.nodes.length-1;i++)
      node.calc_output()
  }
  dump(){
    for(let i=0,node=this.nodes[i];i<this.nodes.length;i++)
      console.dir(node)
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
    console.dir(this.layers)
    for(let i=0;i<layer_count-1;i++){//layer_count=3
      let connections=[];
      for(let j=0;j<this.layers[i].nodes.length;j++){
        let upstream_node=this.layers[i].nodes[j];
        for (let k = 0; k < this.layers[i+1].nodes.length-1; k++) {
          let downstream=this.layers[i+1].nodes[k];
          connections.push(new Connection(upstream_node,downstream))
        }
      }
      for(let i=0,conn=connections[i];i<connections.length;i++){
        this.connections.add_connection(conn)
        conn.downstream_node.append_upstream_connection(conn)
        conn.upstream_node.append_downstream_connection(conn)
      }
    }
  }
  train(labels:any[], data_set:any[], rate:number, epoch:number){
        // for i in range(epoch):
        //     for d in range(len(data_set)):
        //         this.train_one_sample(labels[d], data_set[d], rate)
        //         # print 'sample %d training finished' % d
  }
  train_one_sample(label:any[], sample:any[], rate:number){
        // this.predict(sample)
        // this.calc_delta(label)
        // this.update_weight(rate)
  }
  calc_delta(label:any[]){
        // output_nodes = this.layers[-1].nodes
        // for i in range(len(label)):
        //     output_nodes[i].calc_output_layer_delta(label[i])
        // for layer in this.layers[-2::-1]:
        //     for node in layer.nodes:
        //         node.calc_hidden_layer_delta()
  }
  update_weight(rate:number){
        // for layer in this.layers[:-1]:
        //     for node in layer.nodes:
        //         for conn in node.downstream:
        //             conn.update_weight(rate)
  }
  calc_gradient(){
        // for layer in self.layers[:-1]:
        //     for node in layer.nodes:
        //         for conn in node.downstream:
        //             conn.calc_gradient()
  }
  get_gradient(label:number, sample:number){
        // this.predict(sample)
        // this.calc_delta(label)
        // this.calc_gradient()
  }
  predict(sample:number){
        // this.layers[0].set_output(sample)
        // for i in range(1, len(this.layers)):
        //     this.layers[i].calc_output()
        // return map(lambda node: node.output, this.layers[-1].nodes[:-1])
  }
  dump(){
    // for(let layer as Layer in this.layers){
    //   layer.dump()
    // }
  }
}
class Connections{
  protected conn:Connection[];
  constructor(){this.conn=new Array<Connection>()}
  add_connection(connection:Connection){this.conn.push(connection);}
  dump(){for(let i=0;i<this.conn.length;i++)this.conn[i].toString()}
}
const a=new Network([8,3,8]);
console.log(a)
