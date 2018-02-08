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

import {customGradient} from '../globals';
import * as util from '../util';
import * as axis_util from './axis_util';
import {doc, operation} from './decorators';
import * as ops from './ops';
import {Tensor} from './tensor';

export class Ops {
  /**
   * Computes the softmax normalized vector given the logits.
   * @param logits The logits array.
   * @param dim The dimension softmax would be performed on. Defaults to -1
   *     which indicates the last dimension.
   */
  @doc({heading: 'Operations', subheading: 'Normalization'})
  @operation
  static softmax<T extends Tensor>(logits: T, dim = -1): T {
    if (dim === -1) {
      dim = logits.rank - 1;
    }
    if (dim !== logits.rank - 1) {
      throw Error(
          'Softmax along a non-last dimension is not yet supported. ' +
          `Logits was rank ${logits.rank} and dim was ${dim}`);
    }

    const gradients = (dy: T, y: T) => {
      return {
        logits: () => {
          const dyTimesY = dy.mul(y);
          const keepDims = true;
          return dyTimesY.sub(dyTimesY.sum([dim], keepDims).mul(y));
        }
      };
    };

    return customGradient('softmax', () => {
      // Do it in log space for numerical stability.
      // exp(X - logSumExp(X))
      const keepDims = true;
      const lse = logits.logSumExp([dim], keepDims);
      const logResult = logits.toFloat().sub(lse);
      const value = logResult.exp() as T;
      return {value, gradients};
    }, {logits});
  }

  /**
   * Computes softmax cross entropy between logits and labels.
   *
   * Measures the probability error in discrete classification tasks in which
   * the classes are mutually exclusive (each entry is in exactly one class).
   * For example, each CIFAR-10 image is labeled with one and only one label: an
   * image can be a dog or a truck, but not both.
   *
   * NOTE: While the classes are mutually exclusive, their probabilities need
   * not be. All that is required is that each row of labels is a valid
   * probability distribution. If they are not, the computation of the gradient
   * will be incorrect.
   *
   * WARNING: This op expects unscaled logits, since it performs a softmax on
   * logits internally for efficiency. Do not call this op with the output of
   * softmax, as it will produce incorrect results.
   *
   * logits and labels must have the same shape, e.g. [batch_size, num_classes]
   * and the same dtype.
   * @param labels The labels array.
   * @param logits The logits array.
   * @param dim The dimension softmax would be performed on. Defaults to -1
   *     which indicates the last dimension.
   */
  @doc({
    heading: 'Operations',
    subheading: 'Classification',
    namespace: 'losses'
  })
  @operation
  static softmaxCrossEntropy<T extends Tensor, O extends Tensor>(
      labels: T, logits: T, dim = -1): O {
    util.assertShapesMatch(
        labels.shape, logits.shape, 'Error in softmaxCrossEntropy: ');
    if (dim === -1) {
      dim = logits.rank - 1;
    }
    if (dim !== logits.rank - 1) {
      throw Error(
          `Softmax cross entropy along a non-last dimension is not yet ` +
          `supported. Labels / logits was rank ${logits.rank} ` +
          `and dim was ${dim}`);
    }
    // Use a custom gradient for numerical stability.
    return customGradient('softmaxCrossEntropy', () => {
      const softmaxLogits = logits.softmax(dim);
      const costVector =
          ops.scalar(1e-5).add(softmaxLogits).log().mul(labels).neg();
      const value = costVector.sum([dim]) as O;

      const gradients = (dy: O, y: O) => {
        const dyShape = axis_util.expandShapeToKeepDim(dy.shape, [dim]);

        return {
          logits: () =>
              dy.reshape(dyShape).mul(softmaxLogits.sub(labels.toFloat())),
          labels: () => dy.reshape(dyShape).mul(labels.sub(softmaxLogits))
        };
      };

      return {value, gradients};
    }, {labels, logits});
  }
}
