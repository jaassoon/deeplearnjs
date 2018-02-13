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

import {KernelNode} from '../../tape_types';
import {Tensor} from '../../tensor';
import {Rank} from '../../types';

export interface UnaryNode<R extends Rank, T extends Tensor<R> = Tensor<R>>
    extends KernelNode {
  inputAndArgs: {inputs: {x: T;};};
  output: T;
  gradient: (dy: T, y: T) => {
    x: () => T;
  };
}

export interface LeakyReluNode<R extends Rank, T extends Tensor<R> = Tensor<R>>
    extends KernelNode {
  inputAndArgs: {inputs: {x: T;}; args: {alpha: number;};};
  output: T;
  gradient: (dy: T, y: T) => {
    x: () => T;
  };
}
export interface StepNode<R extends Rank, T extends Tensor<R> = Tensor<R>>
    extends KernelNode {
  inputAndArgs: {inputs: {x: T;}; args: {alpha: number;};};
  output: T;
  gradient: (dy: T, y: T) => {
    x: () => T;
  };
}

export interface ClipNode<R extends Rank, T extends Tensor<R> = Tensor<R>>
    extends KernelNode {
  inputAndArgs: {inputs: {x: T;}; args: {min: number; max: number;};};
  output: T;
  gradient: (dy: T, y: T) => {
    x: () => T;
  };
}

export interface TransposeNode<R extends Rank, T extends Tensor<R> = Tensor<R>>
    extends KernelNode {
  inputAndArgs: {inputs: {x: T;}; args: {perm: number[];};};
  output: T;
  gradient: (dy: T, y: T) => {
    x: () => T;
  };
}

export interface TileNode<R extends Rank, T extends Tensor<R> = Tensor<R>>
    extends KernelNode {
  inputAndArgs: {inputs: {x: T;}; args: {reps: number[];};};
  output: T;
  gradient: (dy: T, y: T) => {
    x: () => T;
  };
}
