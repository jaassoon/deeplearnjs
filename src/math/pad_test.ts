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

// dl.pad1d
{
  const tests = () => {
    it('Should pad 1D arrays', () => {
      const a = dl.tensor1d([1, 2, 3, 4, 5, 6], 'int32');
      const b = dl.pad1d(a, [2, 3]);
      test_util.expectArraysClose(b, [0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0]);
    });

    it('Should not pad 1D arrays with 0s', () => {
      const a = dl.tensor1d([1, 2, 3, 4], 'int32');
      const b = dl.pad1d(a, [0, 0]);
      test_util.expectArraysClose(b, [1, 2, 3, 4]);
    });

    it('Should handle padding with custom value', () => {
      let a = dl.tensor1d([1, 2, 3, 4], 'int32');
      let b = dl.pad1d(a, [2, 3], 9);
      test_util.expectArraysClose(b, [9, 9, 1, 2, 3, 4, 9, 9, 9]);

      a = dl.tensor1d([1, 2, 3, 4]);
      b = dl.pad1d(a, [2, 1], 1.1);
      test_util.expectArraysClose(b, [1.1, 1.1, 1, 2, 3, 4, 1.1]);

      a = dl.tensor1d([1, 2, 3, 4]);
      b = dl.pad1d(a, [2, 1], 1);
      test_util.expectArraysClose(b, [1, 1, 1, 2, 3, 4, 1]);
    });

    it('Should handle NaNs with 1D arrays', () => {
      const a = dl.tensor1d([1, NaN, 2, NaN]);
      const b = dl.pad1d(a, [1, 1]);
      test_util.expectArraysClose(b, [0, 1, NaN, 2, NaN, 0]);
    });

    it('Should handle invalid paddings', () => {
      const a = dl.tensor1d([1, 2, 3, 4], 'int32');
      const f = () => {
        dl.pad1d(a, [2, 2, 2]);
      };
      expect(f).toThrowError();
    });
  };

  test_util.describeMathCPU('pad1d', [tests]);
  test_util.describeMathGPU('pad1d', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}

// dl.path2d
{
  const tests = () => {
    it('Should pad 2D arrays', () => {
      let a = dl.tensor2d([[1], [2]], [2, 1], 'int32');
      let b = dl.pad2d(a, [[1, 1], [1, 1]]);
      // 0, 0, 0
      // 0, 1, 0
      // 0, 2, 0
      // 0, 0, 0
      test_util.expectArraysClose(b, [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0]);

      a = dl.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
      b = dl.pad2d(a, [[2, 2], [1, 1]]);
      // 0, 0, 0, 0, 0
      // 0, 0, 0, 0, 0
      // 0, 1, 2, 3, 0
      // 0, 4, 5, 6, 0
      // 0, 0, 0, 0, 0
      // 0, 0, 0, 0, 0
      test_util.expectArraysClose(b, [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0,
        0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
      ]);
    });

    it('Should not pad 2D arrays with 0s', () => {
      const a = dl.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
      const b = dl.pad2d(a, [[0, 0], [0, 0]]);
      test_util.expectArraysClose(b, [1, 2, 3, 4, 5, 6]);
    });

    it('Should handle padding with custom value', () => {
      let a = dl.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
      let b = dl.pad2d(a, [[1, 1], [1, 1]], 10);
      test_util.expectArraysClose(b, [
        10, 10, 10, 10, 10, 10, 1,  2,  3,  10,
        10, 4,  5,  6,  10, 10, 10, 10, 10, 10
      ]);

      a = dl.tensor2d([[1], [1]], [2, 1]);
      b = dl.pad2d(a, [[1, 1], [1, 1]], -2.1);
      test_util.expectArraysClose(
          b,
          [-2.1, -2.1, -2.1, -2.1, 1, -2.1, -2.1, 1, -2.1, -2.1, -2.1, -2.1]);

      a = dl.tensor2d([[1], [1]], [2, 1]);
      b = dl.pad2d(a, [[1, 1], [1, 1]], -2);
      test_util.expectArraysClose(
          b, [-2, -2, -2, -2, 1, -2, -2, 1, -2, -2, -2, -2]);
    });

    it('Should handle NaNs with 2D arrays', () => {
      const a = dl.tensor2d([[1, NaN], [1, NaN]], [2, 2]);
      const b = dl.pad2d(a, [[1, 1], [1, 1]]);
      // 0, 0, 0,   0
      // 0, 1, NaN, 0
      // 0, 1, NaN, 0
      // 0, 0, 0,   0
      test_util.expectArraysClose(
          b, [0, 0, 0, 0, 0, 1, NaN, 0, 0, 1, NaN, 0, 0, 0, 0, 0]);
    });

    it('Should handle invalid paddings', () => {
      const a = dl.tensor2d([[1], [2]], [2, 1], 'int32');
      const f = () => {
        dl.pad2d(a, [[2, 2, 2], [1, 1, 1]]);
      };
      expect(f).toThrowError();
    });
  };

  test_util.describeMathCPU('pad2d', [tests]);
  test_util.describeMathGPU('pad2d', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
