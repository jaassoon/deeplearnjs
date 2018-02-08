/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as util from '../../util';
import * as ops from '../ops';
import {Tensor} from '../tensor';
import {Rank} from '../types';
import {MathBackend} from './backend';
import {ArgMaxNode, ArgMinNode} from './types/argminmax';
// tslint:disable-next-line:max-line-length
import {BatchNorm2DNode, BatchNorm3DNode, BatchNorm4DNode} from './types/batchnorm';
import {BinaryNode} from './types/binary';
import {CastNode} from './types/cast';
// tslint:disable-next-line:max-line-length
import {ConcatNode} from './types/concat';
// tslint:disable-next-line:max-line-length
import {Conv2DDerBiasNode, Conv2DDerFilterNode, Conv2DDerInputNode, Conv2DNode, DepthwiseConv2DNode} from './types/conv';
import {GatherNode} from './types/gather';
import {EqualNode, LogicalNode, WhereNode} from './types/logical';
import {LRN4DNode} from './types/lrn';
import {MatMulNode} from './types/matmul';
import {MaximumNode, MaxNode, MinimumNode, MinNode} from './types/minmax';
import {MultinomialNode} from './types/multinomial';
import {OneHotNode} from './types/onehot';
import {Pad1DNode, Pad2DNode} from './types/pad';
// tslint:disable-next-line:max-line-length
import {PoolBackpropNode, PoolNode} from './types/pool';
import {PowNode} from './types/pow';
import {PReLUNode} from './types/prelu';
import {ReshapeNode} from './types/reshape';
import {ResizeBilinearNode} from './types/resize_bilinear';
import {Reverse4DNode} from './types/reverse';
// tslint:disable-next-line:max-line-length
import {Slice1DNode, Slice2DNode, Slice3DNode, Slice4DNode} from './types/slice';
import {SumNode} from './types/sum';
import {TopKIndicesNode, TopKValuesNode} from './types/topk';
// tslint:disable-next-line:max-line-length
import {ClipNode, LeakyReluNode, StepNode, TileNode, TransposeNode, UnaryNode} from './types/unary';

export function
executeKernel<R extends Rank, K extends keyof KernelConfigRegistry<R>, O extends
                  KernelConfigRegistry<R>[K]['output']>(
    backend: MathBackend, kernelName: K,
    inputAndArgs: KernelConfigRegistry<R>[K]['inputAndArgs']): O {
  if (kernelName === 'MatMul') {
    const config = inputAndArgs as MatMulNode['inputAndArgs'];
    return backend.matMul(
               config.inputs.a, config.inputs.b, config.args.aOrientation,
               config.args.bOrientation) as O;
  } else if (kernelName === 'Slice1D') {
    const config = inputAndArgs as Slice1DNode['inputAndArgs'];
    return backend.slice1D(
               config.inputs.x, config.args.begin, config.args.size) as O;
  } else if (kernelName === 'Slice2D') {
    const config = inputAndArgs as Slice2DNode['inputAndArgs'];
    return backend.slice2D(
               config.inputs.x, config.args.begin, config.args.size) as O;
  } else if (kernelName === 'Slice3D') {
    const config = inputAndArgs as Slice3DNode['inputAndArgs'];
    return backend.slice3D(
               config.inputs.x, config.args.begin, config.args.size) as O;
  } else if (kernelName === 'Slice4D') {
    const config = inputAndArgs as Slice4DNode['inputAndArgs'];
    return backend.slice4D(
               config.inputs.x, config.args.begin, config.args.size) as O;
  } else if (kernelName === 'Reverse4D') {
    const config = inputAndArgs as Reverse4DNode['inputAndArgs'];
    return backend.reverse4D(config.inputs.x, config.args.axis) as O;
  } else if (kernelName === 'Concat') {
    const config = inputAndArgs as ConcatNode['inputAndArgs'];
    return backend.concat(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'Neg') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.neg(config.inputs.x) as O;
  } else if (kernelName === 'Add') {
    const config = inputAndArgs as BinaryNode['inputAndArgs'];
    return backend.add(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'Sub') {
    const config = inputAndArgs as BinaryNode['inputAndArgs'];
    return backend.subtract(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'Mul') {
    const config = inputAndArgs as BinaryNode['inputAndArgs'];
    return backend.multiply(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'Div') {
    const config = inputAndArgs as BinaryNode['inputAndArgs'];
    return backend.divide(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'Sum') {
    const config = inputAndArgs as SumNode['inputAndArgs'];
    return backend.sum(config.inputs.x, config.args.axes) as O;
  } else if (kernelName === 'ArgMax') {
    const config = inputAndArgs as ArgMaxNode['inputAndArgs'];
    return backend.argMax(config.inputs.x, config.args.axes) as O;
  } else if (kernelName === 'ArgMin') {
    const config = inputAndArgs as ArgMinNode['inputAndArgs'];
    return backend.argMin(config.inputs.x, config.args.axes) as O;
  } else if (kernelName === 'Equal') {
    const config = inputAndArgs as EqualNode['inputAndArgs'];
    return backend.equal(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'NotEqual') {
    const config = inputAndArgs as EqualNode['inputAndArgs'];
    return backend.notEqual(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'Less') {
    const config = inputAndArgs as EqualNode['inputAndArgs'];
    return backend.less(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'LessEqual') {
    const config = inputAndArgs as EqualNode['inputAndArgs'];
    return backend.lessEqual(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'Greater') {
    const config = inputAndArgs as EqualNode['inputAndArgs'];
    return backend.greater(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'GreaterEqual') {
    const config = inputAndArgs as EqualNode['inputAndArgs'];
    return backend.greaterEqual(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'LogicalNot') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.logicalNot(config.inputs.x) as O;
  } else if (kernelName === 'LogicalAnd') {
    const config = inputAndArgs as LogicalNode['inputAndArgs'];
    return backend.logicalAnd(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'LogicalOr') {
    const config = inputAndArgs as LogicalNode['inputAndArgs'];
    return backend.logicalOr(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'LogicalXor') {
    const config = inputAndArgs as LogicalNode['inputAndArgs'];
    return backend.logicalXor(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'Where') {
    const config = inputAndArgs as WhereNode['inputAndArgs'];
    return backend.where(
               config.inputs.condition, config.inputs.a, config.inputs.b,
               config.args.dtype) as O;
  } else if (kernelName === 'TopKValues') {
    const config = inputAndArgs as TopKValuesNode<R>['inputAndArgs'];
    return backend.topKValues(config.inputs.x, config.args.k) as O;
  } else if (kernelName === 'TopKIndices') {
    const config = inputAndArgs as TopKIndicesNode['inputAndArgs'];
    return backend.topKIndices(config.inputs.x, config.args.k) as O;
  } else if (kernelName === 'Min') {
    const config = inputAndArgs as MinNode['inputAndArgs'];
    return backend.min(config.inputs.x, config.args.axes) as O;
  } else if (kernelName === 'Minimum') {
    const config = inputAndArgs as MinimumNode['inputAndArgs'];
    return backend.minimum(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'Max') {
    const config = inputAndArgs as MaxNode['inputAndArgs'];
    return backend.max(config.inputs.x, config.args.axes) as O;
  } else if (kernelName === 'Maximum') {
    const config = inputAndArgs as MaximumNode['inputAndArgs'];
    return backend.maximum(config.inputs.a, config.inputs.b) as O;
  } else if (kernelName === 'Ceil') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.ceil(config.inputs.x) as O;
  } else if (kernelName === 'Floor') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.floor(config.inputs.x) as O;
  } else if (kernelName === 'Pow') {
    const config = inputAndArgs as PowNode<R>['inputAndArgs'];
    return backend.pow(config.inputs.base, config.inputs.exp) as O;
  } else if (kernelName === 'Exp') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.exp(config.inputs.x) as O;
  } else if (kernelName === 'Log') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.log(config.inputs.x) as O;
  } else if (kernelName === 'Sqrt') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.sqrt(config.inputs.x) as O;
  } else if (kernelName === 'Square') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.square(config.inputs.x) as O;
  } else if (kernelName === 'Relu') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.relu(config.inputs.x) as O;
  } else if (kernelName === 'Reshape') {
    const config = inputAndArgs as ReshapeNode['inputAndArgs'];
    const x = config.inputs.x;
    const newShape = config.args.newShape;
    return Tensor.make(newShape, {dataId: x.dataId}, x.dtype) as O;
  } else if (kernelName === 'Cast') {
    const config = inputAndArgs as CastNode['inputAndArgs'];
    const x = config.inputs.x;
    const newDType = config.args.newDType;

    if (!util.hasEncodingLoss(x.dtype, newDType)) {
      // We don't change the underlying data, since we cast to higher
      // precision.
      return Tensor.make(x.shape, {dataId: x.dataId}, newDType) as O;
    }
    if (newDType === 'int32') {
      return backend.int(x) as O;
    } else if (newDType === 'bool') {
      return backend.notEqual(x, ops.scalar(0, x.dtype)) as O;
    } else {
      throw new Error(`Error in Cast: unknown dtype argument (${newDType})`);
    }
  } else if (kernelName === 'LeakyRelu') {
    const config = inputAndArgs as LeakyReluNode<R>['inputAndArgs'];
    return backend.leakyRelu(config.inputs.x, config.args.alpha) as O;
  } else if (kernelName === 'PReLU') {
    const config = inputAndArgs as PReLUNode<R>['inputAndArgs'];
    return backend.prelu(config.inputs.x, config.inputs.alpha) as O;
  } else if (kernelName === 'PReLUDer') {
    const config = inputAndArgs as PReLUNode<R>['inputAndArgs'];
    return backend.preluDer(config.inputs.x, config.inputs.alpha) as O;
  } else if (kernelName === 'Elu') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.elu(config.inputs.x) as O;
  } else if (kernelName === 'EluDer') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.eluDer(config.inputs.x) as O;
  } else if (kernelName === 'Selu') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.selu(config.inputs.x) as O;
  } else if (kernelName === 'Abs') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.abs(config.inputs.x) as O;
  } else if (kernelName === 'Sigmoid') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.sigmoid(config.inputs.x) as O;
  } else if (kernelName === 'Step') {
    const config = inputAndArgs as StepNode<R>['inputAndArgs'];
    return backend.step(config.inputs.x, config.args.alpha) as O;
  } else if (kernelName === 'Sin') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.sin(config.inputs.x) as O;
  } else if (kernelName === 'Cos') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.cos(config.inputs.x) as O;
  } else if (kernelName === 'Tan') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.tan(config.inputs.x) as O;
  } else if (kernelName === 'Asin') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.asin(config.inputs.x) as O;
  } else if (kernelName === 'Acos') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.acos(config.inputs.x) as O;
  } else if (kernelName === 'Atan') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.atan(config.inputs.x) as O;
  } else if (kernelName === 'Sinh') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.sinh(config.inputs.x) as O;
  } else if (kernelName === 'Cosh') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.cosh(config.inputs.x) as O;
  } else if (kernelName === 'Tanh') {
    const config = inputAndArgs as UnaryNode<R>['inputAndArgs'];
    return backend.tanh(config.inputs.x) as O;
  } else if (kernelName === 'Clip') {
    const config = inputAndArgs as ClipNode<R>['inputAndArgs'];
    return backend.clip(config.inputs.x, config.args.min, config.args.max) as O;
  } else if (kernelName === 'Tile') {
    const config = inputAndArgs as TileNode<R>['inputAndArgs'];
    return backend.tile(config.inputs.x, config.args.reps) as O;
  } else if (kernelName === 'Gather') {
    const config = inputAndArgs as GatherNode<R>['inputAndArgs'];
    return backend.gather(
               config.inputs.x, config.inputs.indices, config.args.axis) as O;
  } else if (kernelName === 'Pad1D') {
    const config = inputAndArgs as Pad1DNode['inputAndArgs'];
    return backend.pad1D(
               config.inputs.x, config.args.paddings,
               config.args.constantValue) as O;
  } else if (kernelName === 'Pad2D') {
    const config = inputAndArgs as Pad2DNode['inputAndArgs'];
    return backend.pad2D(
               config.inputs.x, config.args.paddings,
               config.args.constantValue) as O;
  } else if (kernelName === 'Transpose') {
    const config = inputAndArgs as TransposeNode<R>['inputAndArgs'];
    return backend.transpose(config.inputs.x, config.args.perm) as O;
  } else if (kernelName === 'Conv2D') {
    const config = inputAndArgs as Conv2DNode['inputAndArgs'];
    return backend.conv2d(
               config.inputs.x, config.inputs.filter, config.inputs.bias,
               config.args.convInfo) as O;
  } else if (kernelName === 'Conv2DDerInput') {
    const config = inputAndArgs as Conv2DDerInputNode['inputAndArgs'];
    return backend.conv2dDerInput(
               config.inputs.dy, config.inputs.filter, config.args.convInfo) as
        O;
  } else if (kernelName === 'Conv2DDerFilter') {
    const config = inputAndArgs as Conv2DDerFilterNode['inputAndArgs'];
    return backend.conv2dDerFilter(
               config.inputs.x, config.inputs.dy, config.args.convInfo) as O;
  } else if (kernelName === 'Conv2DDerBias') {
    const config = inputAndArgs as Conv2DDerBiasNode['inputAndArgs'];
    return backend.conv2dDerBias(config.inputs.dy) as O;
  } else if (kernelName === 'DepthwiseConv2D') {
    const config = inputAndArgs as DepthwiseConv2DNode['inputAndArgs'];
    return backend.depthwiseConv2D(
               config.inputs.x, config.inputs.filter, config.args.convInfo) as
        O;
  } else if (kernelName === 'MaxPool') {
    const config = inputAndArgs as PoolNode['inputAndArgs'];
    return backend.maxPool(config.inputs.x, config.args.convInfo) as O;
  } else if (kernelName === 'MaxPoolBackprop') {
    const config = inputAndArgs as PoolBackpropNode['inputAndArgs'];
    return backend.maxPoolBackprop(
               config.inputs.dy, config.inputs.x, config.args.convInfo) as O;
  } else if (kernelName === 'AvgPool') {
    const config = inputAndArgs as PoolNode['inputAndArgs'];
    return backend.avgPool(config.inputs.x, config.args.convInfo) as O;
  } else if (kernelName === 'AvgPoolBackprop') {
    const config = inputAndArgs as PoolBackpropNode['inputAndArgs'];
    return backend.avgPoolBackprop(
               config.inputs.dy, config.inputs.x, config.args.convInfo) as O;
  } else if (kernelName === 'MinPool') {
    const config = inputAndArgs as PoolNode['inputAndArgs'];
    return backend.minPool(config.inputs.x, config.args.convInfo) as O;
  } else if (kernelName === 'ResizeBilinear') {
    const config = inputAndArgs as ResizeBilinearNode['inputAndArgs'];
    return backend.resizeBilinear(
               config.inputs.x, config.args.newHeight, config.args.newWidth,
               config.args.alignCorners) as O;
  } else if (kernelName === 'BatchNorm4D') {
    const config = inputAndArgs as BatchNorm4DNode['inputAndArgs'];
    return backend.batchNormalization4D(
               config.inputs.x, config.inputs.mean, config.inputs.variance,
               config.args.varianceEpsilon, config.inputs.scale,
               config.inputs.offset) as O;
  } else if (kernelName === 'BatchNorm3D') {
    const config = inputAndArgs as BatchNorm3DNode['inputAndArgs'];
    return backend.batchNormalization3D(
               config.inputs.x, config.inputs.mean, config.inputs.variance,
               config.args.varianceEpsilon, config.inputs.scale,
               config.inputs.offset) as O;
  } else if (kernelName === 'BatchNorm2D') {
    const config = inputAndArgs as BatchNorm2DNode['inputAndArgs'];
    return backend.batchNormalization2D(
               config.inputs.x, config.inputs.mean, config.inputs.variance,
               config.args.varianceEpsilon, config.inputs.scale,
               config.inputs.offset) as O;
  } else if (kernelName === 'LRN4D') {
    const config = inputAndArgs as LRN4DNode['inputAndArgs'];
    return backend.localResponseNormalization4D(
               config.inputs.x, config.args.radius, config.args.bias,
               config.args.alpha, config.args.beta, config.args.normRegion) as
        O;
  } else if (kernelName === 'Multinomial') {
    const config = inputAndArgs as MultinomialNode['inputAndArgs'];
    return backend.multinomial(
               config.inputs.probs, config.args.numSamples, config.args.seed) as
        O;
  } else if (kernelName === 'OneHot') {
    const config = inputAndArgs as OneHotNode['inputAndArgs'];
    return backend.oneHot(
               config.inputs.indices, config.args.depth, config.args.onValue,
               config.args.offValue) as O;
  }
  throw new Error(`No backend method found for kernel ${kernelName}`);
}

export interface KernelConfigRegistry<R extends Rank> {
  MatMul: MatMulNode;
  Slice1D: Slice1DNode;
  Slice2D: Slice2DNode;
  Slice3D: Slice3DNode;
  Slice4D: Slice4DNode;
  Reverse4D: Reverse4DNode;
  Concat: ConcatNode;
  Neg: UnaryNode<R>;
  Add: BinaryNode;
  Sub: BinaryNode;
  Mul: BinaryNode;
  Div: BinaryNode;
  Sum: SumNode;
  ArgMax: ArgMaxNode;
  ArgMin: ArgMinNode;
  Equal: EqualNode;
  NotEqual: EqualNode;
  Less: EqualNode;
  LessEqual: EqualNode;
  Greater: EqualNode;
  GreaterEqual: EqualNode;
  LogicalNot: UnaryNode<R>;
  LogicalAnd: LogicalNode;
  LogicalOr: LogicalNode;
  LogicalXor: LogicalNode;
  Where: WhereNode;
  TopKValues: TopKValuesNode<R>;
  TopKIndices: TopKIndicesNode;
  Min: MinNode;
  Minimum: MinimumNode;
  Max: MaxNode;
  Maximum: MaximumNode;
  Ceil: UnaryNode<R>;
  Floor: UnaryNode<R>;
  Pow: PowNode<R>;
  Exp: UnaryNode<R>;
  Log: UnaryNode<R>;
  Sqrt: UnaryNode<R>;
  Square: UnaryNode<R>;
  Relu: UnaryNode<R>;
  LeakyRelu: LeakyReluNode<R>;
  PReLU: PReLUNode<R>;
  PReLUDer: PReLUNode<R>;
  Reshape: ReshapeNode;
  Cast: CastNode;
  Elu: UnaryNode<R>;
  EluDer: UnaryNode<R>;
  Selu: UnaryNode<R>;
  Abs: UnaryNode<R>;
  Sigmoid: UnaryNode<R>;
  Step: StepNode<R>;
  Sin: UnaryNode<R>;
  Cos: UnaryNode<R>;
  Tan: UnaryNode<R>;
  Asin: UnaryNode<R>;
  Acos: UnaryNode<R>;
  Atan: UnaryNode<R>;
  Sinh: UnaryNode<R>;
  Cosh: UnaryNode<R>;
  Tanh: UnaryNode<R>;
  Clip: ClipNode<R>;
  Transpose: TransposeNode<R>;
  Pad1D: Pad1DNode;
  Pad2D: Pad2DNode;
  Tile: TileNode<R>;
  Gather: GatherNode<R>;
  Conv2D: Conv2DNode;
  Conv2DDerInput: Conv2DDerInputNode;
  Conv2DDerFilter: Conv2DDerFilterNode;
  Conv2DDerBias: Conv2DDerBiasNode;
  DepthwiseConv2D: Conv2DNode;
  MaxPool: PoolNode;
  MaxPoolBackprop: PoolBackpropNode;
  AvgPool: PoolNode;
  AvgPoolBackprop: PoolBackpropNode;
  MinPool: PoolNode;
  ResizeBilinear: ResizeBilinearNode;
  BatchNorm4D: BatchNorm4DNode;
  BatchNorm3D: BatchNorm3DNode;
  BatchNorm2D: BatchNorm2DNode;
  LRN4D: LRN4DNode;
  Multinomial: MultinomialNode;
  OneHot: OneHotNode;
}
export type Kernel = keyof KernelConfigRegistry<Rank>;
