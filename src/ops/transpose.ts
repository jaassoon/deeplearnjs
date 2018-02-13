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

import {operation} from './operation';
import {doc} from '../doc';
import {ENV} from '../environment';
import {Tensor} from '../tensor';
import {Rank} from '../types';
import * as util from '../util';
import * as axis_util from './axis_util';

export class Ops {
  /**
   * Transposes the array. Permutes the dimensions according to `perm`.
   *
   * The returned array's dimension `i` will correspond to the input dimension
   * `perm[i]`. If `perm` is not given, it is set to `[n-1...0]`, where `n` is
   * the rank of the input array. Hence by default, this operation performs a
   * regular matrix transpose on 2-D input arrays.
   *
   * @param x The array to transpose.
   * @param perm Optional. The permutation of the dimensions of a.
   */
  @doc({heading: 'Operations', subheading: 'Matrices'})
  @operation
  static transpose<R extends Rank>(x: Tensor<R>, perm?: number[]): Tensor<R> {
    if (perm == null) {
      perm = x.shape.map((s, i) => i).reverse();
    }
    const der = (dy: Tensor) => {
      const undoPerm = axis_util.getUndoAxesPermutation(perm);
      const derX = () => dy.transpose(undoPerm);
      return {x: derX};
    };
    util.assert(
        x.rank === perm.length,
        `Error in transpose: rank of input ${x.rank} ` +
            `must match length of perm ${perm}.`);
    return ENV.engine.executeKernel(
               'Transpose', {inputs: {x}, args: {perm}}, der) as Tensor<R>;
  }
}
