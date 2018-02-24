import {MnistData} from './data';
import * as fc from '../full_conn/full_conn';
import * as dl from 'deeplearn';

let data: MnistData;
async function load() {
  data = new MnistData();
  await data.load();
}
async function mnist() {
  await load();
  const network=new fc.Network([784,100,10])
  const datas=[] as dl.Tensor2D[];
  const labels=[] as dl.Tensor2D[];
  for(let i=0;i<256;i++){
    datas.push(data.trainingData[0][i].reshape([784,1]))
    labels.push(data.trainingData[1][i].reshape([10,1]))
  }
  network.train(labels,datas,.01,1)
}
mnist();