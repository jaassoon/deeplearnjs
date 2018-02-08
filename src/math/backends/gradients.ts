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

import {ENV} from '../../environment';
import {tidy} from '../../globals';
import * as util from '../../util';
import {doc} from '../decorators';
import {Scalar, Tensor, Variable} from '../tensor';
import {NamedTensorMap, Rank} from '../types';
import {CustomGradientFunc} from './backend_engine';
import {ScopeFn, ScopeResult} from './tape_util';

export class Gradients {
  /**
   * Create a new gradients scope. Similar to scope, but forces all inner scopes
   * to not clean up so that gradient operations can be used inside of this
   * scope.
   * @param nameOrScopeFn The name of the scope, or the function to execute.
   *     If a name is provided, the 2nd argument should be the function.
   *     If a name is provided, and debug mode is on, the timing and the memory
   *     usage of the function will be tracked and displayed on the console
   *     using the provided name.
   * @param scopeFn The function to execute.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static gradientsScope<T extends ScopeResult>(
      nameOrScopeFn: string|ScopeFn<T>, scopeFn?: ScopeFn<T>): T {
    return tidy(nameOrScopeFn, scopeFn, true /* gradientsScope */);
  }

  /**
   * Computes and returns the vector jacobian product of f(x) with respect to x.
   * This method allows you to provide a non-scalar dy to backprop from.
   *
   * @param f The function to execute. f() should return a Tensor of the same
   * shape and dtype as dy.
   * @param x The input to compute dy/dx over. This can be a single value or
   * an object mapping a string to a Tensor. If using the object mode, this
   * method will return an object of the same shape.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static vjp<T extends Tensor|NamedTensorMap, R extends Rank>(
      f: () => Tensor<R>, x: T, dy: Tensor<R>): T {
    const res = Gradients.valueAndGradients(f, x, dy);
    res.value.dispose();
    return res.gradients;
  }

  /**
   * Computes and returns the gradient of f(x) with respect to x.
   *
   * @param f The function to execute. f() should return a Tensor.
   * @param x The input to compute de/dx over. This can be a single value or
   * an object mapping a string to a Tensor. If using the object mode, this
   * method will return an object of the same shape.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static gradients<R extends Rank, T extends Tensor|NamedTensorMap>(
      f: () => Tensor<R>, x: T): T {
    const res = Gradients.valueAndGradients(f, x);
    res.value.dispose();
    return res.gradients;
  }

  /**
   * Computes and returns the gradient of f(x) with respect to the list of
   * trainable variables provided by `varList`. If no list is provided, it
   * defaults to all trainable variables.
   * @param f The function to execute. f() should return a scalar.
   * @param varList An optional list of variables to provide gradients with
   * respect to. Defaults to all trainable variables.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static variableGradients(f: () => Scalar, varList?: Variable[]):
      {value: Scalar, gradients: NamedTensorMap} {
    if (varList == null) {
      // Get all of the trainable variables.
      varList = [];
      for (const varName in ENV.engine.registeredVariables) {
        varList.push(ENV.engine.registeredVariables[varName]);
      }
    }
    // Prune non-trainable variables.
    varList = varList.filter(variable => variable.trainable);
    const {value, gradients} = ENV.engine.gradients(f, varList);
    if (value.rank > 0) {
      throw new Error(
          `The user-provided function must return a Scalar, but it ` +
          `returned a rank-${value.rank} tensor`);
    }
    const namedGrads: NamedTensorMap = {};
    varList.forEach((v, i) => {
      if (gradients[i] != null) {
        namedGrads[v.name] = gradients[i];
      }
    });
    return {value, gradients: namedGrads};
  }

  /**
   * Computes and returns the gradient of f(x) with respect to x. Returns
   * both f(x) and f'(x).
   *
   * @param f The function to execute. f() should return a Tensor.
   * @param x The input to compute de/dx over. This can be a single value or
   * an object mapping a string to a Tensor. If using the object mode,
   * this method will return an object of the same shape.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static valueAndGradients<R extends Rank, T extends Tensor|NamedTensorMap>(
      f: () => Tensor<R>, x: T, dy?: Tensor<R>):
      {value: Tensor<R>, gradients: T} {
    const keys = x instanceof Tensor ? null : Object.keys(x);
    const xs = util.flattenNameArrayMap(x, keys);

    const {value, gradients} = ENV.engine.gradients(f, xs, dy);
    const numNullGradients = gradients.filter(g => g == null).length;
    if (numNullGradients > 0) {
      throw new Error(
          `Cannot compute gradient: y is not a function of xs.` +
          `Make sure the xs you are computing gradients with respect ` +
          `to are used inside the gradient function.`);
    }
    const resGradients = (x instanceof Tensor) ?
        gradients[0] as T :
        util.unflattenToNameArrayMap(keys, gradients) as T;
    return {value, gradients: resGradients};
  }

  /**
   * Evaluates a function f() with a custom gradient function f'() to use during
   * backpropagation.
   *
   * @param f The function to evaluate in forward mode. Returns a value Tensor
   *    and a gradient function closure.
   * @param inputs The inputs to compute the gradient with respect to. These
   *    Tensors should be used in f().
   * @param name An optional name for the customGradient method. Used for
   *    debugging.
   */
  @doc({heading: 'Training', subheading: 'Gradients'})
  static customGradient<T extends Tensor>(
      name: string, f: CustomGradientFunc<T>, inputs: NamedTensorMap): T {
    name = name || '';
    return ENV.engine.customGradient(name, f, inputs);
  }
}
