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

import * as dl from '../index';
import * as test_util from '../test_util';

const commonTests = () => {
  it('returns a tensor with the same shape and value', () => {
    const a = dl.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
    const aPrime = dl.clone(a);
    expect(aPrime.shape).toEqual(a.shape);
    test_util.expectArraysClose(aPrime, a);
  });
};

test_util.describeMathCPU('clone', [commonTests]);
test_util.describeMathGPU('clone', [commonTests], [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
