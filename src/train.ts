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

// So typings can propagate.
import {AdadeltaOptimizer} from './math/optimizers/adadelta_optimizer';
import {AdagradOptimizer} from './math/optimizers/adagrad_optimizer';
import {MomentumOptimizer} from './math/optimizers/momentum_optimizer';
import {OptimizerConstructors} from './math/optimizers/optimizer_constructors';
import {SGDOptimizer} from './math/optimizers/sgd_optimizer';

// tslint:disable-next-line:no-unused-expression
[MomentumOptimizer, SGDOptimizer, AdadeltaOptimizer, AdagradOptimizer];

export const train = {
  sgd: OptimizerConstructors.sgd,
  momentum: OptimizerConstructors.momentum,
  adadelta: OptimizerConstructors.adadelta,
  adagrad: OptimizerConstructors.adagrad
};
