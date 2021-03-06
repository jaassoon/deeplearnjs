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

import {BackendType, ENV, Environment, Features} from './environment';
import {MathBackendCPU} from './math/backends/backend_cpu';
import {MathBackendWebGL} from './math/backends/backend_webgl';
import {Tensor} from './math/tensor';
import {DataType, TypedArray} from './math/types';
import * as util from './util';

export type Tests = () => void;

/** Accuracy for tests. */
// TODO(nsthorat || smilkov): Fix this low precision for byte-backed
// textures.
export const TEST_EPSILON = 1e-2;

export function mean(values: TypedArray|number[]) {
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
  }
  return sum / values.length;
}

export function standardDeviation(values: TypedArray|number[], mean: number) {
  let squareDiffSum = 0;
  for (let i = 0; i < values.length; i++) {
    const diff = values[i] - mean;
    squareDiffSum += diff * diff;
  }
  return Math.sqrt(squareDiffSum / values.length);
}

export function kurtosis(values: TypedArray|number[]) {
  // https://en.wikipedia.org/wiki/Kurtosis
  const valuesMean = mean(values);
  const n = values.length;
  let sum2 = 0;
  let sum4 = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i] - valuesMean;
    sum2 += Math.pow(v, 2);
    sum4 += Math.pow(v, 4);
  }
  return (1 / n) * sum4 / Math.pow((1 / n) * sum2, 2);
}

export function skewness(values: TypedArray|number[]) {
  // https://en.wikipedia.org/wiki/Skewness
  const valuesMean = mean(values);
  const n = values.length;
  let sum2 = 0;
  let sum3 = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i] - valuesMean;
    sum2 += Math.pow(v, 2);
    sum3 += Math.pow(v, 3);
  }
  return (1 / n) * sum3 / Math.pow((1 / (n - 1)) * sum2, 3 / 2);
}

export function jarqueBeraNormalityTest(a: Tensor|TypedArray|number[]) {
  let values: TypedArray|number[];
  if (a instanceof Tensor) {
    values = a.dataSync();
  } else {
    values = a;
  }
  // https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test
  const n = values.length;
  const s = skewness(values);
  const k = kurtosis(values);
  const jb = n / 6 * (Math.pow(s, 2) + 0.25 * Math.pow(k - 3, 2));
  // JB test requires 2-degress of freedom from Chi-Square @ 0.95:
  // http://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
  const CHI_SQUARE_2DEG = 5.991;
  if (jb > CHI_SQUARE_2DEG) {
    throw new Error(`Invalid p-value for JB: ${jb}`);
  }
}

export function expectArrayInMeanStdRange(
    actual: Tensor|TypedArray|number[], expectedMean: number,
    expectedStdDev: number, epsilon = TEST_EPSILON) {
  let actualValues: TypedArray|number[];
  if (actual instanceof Tensor) {
    actualValues = actual.dataSync();
  } else {
    actualValues = actual;
  }
  const actualMean = mean(actualValues);
  expectNumbersClose(actualMean, expectedMean, epsilon);
  expectNumbersClose(
      standardDeviation(actualValues, actualMean), expectedStdDev, epsilon);
}

export function expectArraysClose(
    actual: Tensor|TypedArray|number[],
    expected: Tensor|TypedArray|number[]|boolean[], epsilon = TEST_EPSILON) {
  if (!(actual instanceof Tensor) && !(expected instanceof Tensor)) {
    const aType = actual.constructor.name;
    const bType = expected.constructor.name;

    if (aType !== bType) {
      throw new Error(
          `Arrays are of different type actual: ${aType} ` +
          `vs expected: ${bType}`);
    }
  } else if (actual instanceof Tensor && expected instanceof Tensor) {
    if (actual.dtype !== expected.dtype) {
      throw new Error(
          `Arrays are of different type actual: ${actual.dtype} ` +
          `vs expected: ${expected.dtype}.`);
    }
    if (!util.arraysEqual(actual.shape, expected.shape)) {
      throw new Error(
          `Arrays are of different shape actual: ${actual.shape} ` +
          `vs expected: ${expected.shape}.`);
    }
  }

  let actualValues: TypedArray|number[];
  let expectedValues: TypedArray|number[]|boolean[];
  if (actual instanceof Tensor) {
    actualValues = actual.dataSync();
  } else {
    actualValues = actual;
  }
  if (expected instanceof Tensor) {
    expectedValues = expected.dataSync();
  } else {
    expectedValues = expected;
  }

  if (actualValues.length !== expectedValues.length) {
    throw new Error(
        `Arrays have different lengths actual: ${actualValues.length} vs ` +
        `expected: ${expectedValues.length}.\n` +
        `Actual:   ${actualValues}.\n` +
        `Expected: ${expectedValues}.`);
  }
  for (let i = 0; i < expectedValues.length; ++i) {
    const a = actualValues[i];
    const e = expectedValues[i];

    if (!areClose(a, Number(e), epsilon)) {
      throw new Error(
          `Arrays differ: actual[${i}] = ${a}, expected[${i}] = ${e}.\n` +
          `Actual:   ${actualValues}.\n` +
          `Expected: ${expectedValues}.`);
    }
  }
}

export function expectArraysEqual(
    actual: Tensor|TypedArray|number[],
    expected: Tensor|TypedArray|number[]|boolean[]) {
  return expectArraysClose(actual, expected, 0);
}

export function expectNumbersClose(
    a: number, e: number, epsilon = TEST_EPSILON) {
  if (!areClose(a, e, epsilon)) {
    throw new Error(`Numbers differ: actual === ${a}, expected === ${e}`);
  }
}

function areClose(a: number, e: number, epsilon: number): boolean {
  if (isNaN(a) && isNaN(e)) {
    return true;
  }
  if (isNaN(a) || isNaN(e) || Math.abs(a - e) > epsilon) {
    return false;
  }
  return true;
}

export function expectValuesInRange(
    actual: Tensor|TypedArray|number[], low: number, high: number) {
  let actualVals: TypedArray|number[];
  if (actual instanceof Tensor) {
    actualVals = actual.dataSync();
  } else {
    actualVals = actual;
  }
  for (let i = 0; i < actualVals.length; i++) {
    if (actualVals[i] < low || actualVals[i] > high) {
      throw new Error(
          `Value out of range:${actualVals[i]} low: ${low}, high: ${high}`);
    }
  }
}

export function randomArrayInRange(
    n: number, minValue: number, maxValue: number): Float32Array {
  const v = new Float32Array(n);
  const range = maxValue - minValue;
  for (let i = 0; i < n; ++i) {
    v[i] = (Math.random() * range) + minValue;
  }
  return v;
}

export function makeIdentity(n: number): Float32Array {
  const i = new Float32Array(n * n);
  for (let j = 0; j < n; ++j) {
    i[(j * n) + j] = 1;
  }
  return i;
}

export function cpuMultiplyMatrix(
    a: Float32Array, aRow: number, aCol: number, b: Float32Array, bRow: number,
    bCol: number) {
  const result = new Float32Array(aRow * bCol);
  for (let r = 0; r < aRow; ++r) {
    const aOffset = (r * aCol);
    const cOffset = (r * bCol);
    for (let c = 0; c < bCol; ++c) {
      let d = 0;
      for (let k = 0; k < aCol; ++k) {
        d += a[aOffset + k] * b[(k * bCol) + c];
      }
      result[cOffset + c] = d;
    }
  }
  return result;
}

export function cpuDotProduct(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error('cpuDotProduct: incompatible vectors.');
  }
  let d = 0;
  for (let i = 0; i < a.length; ++i) {
    d += a[i] * b[i];
  }
  return d;
}

export function describeMathCPU(
    name: string, tests: Tests[], featuresList?: Features[]) {
  const testNameBase = 'CPU: ' + name;
  describeWithFeaturesAndExecutor(
      testNameBase, tests,
      (testName, tests, features) =>
          executeMathTests(testName, tests, 'cpu', features),
      featuresList);
}

export function describeMathGPU(
    name: string, tests: Tests[], featuresList?: Features[]) {
  const testNameBase = 'WebGL: ' + name;
  describeWithFeaturesAndExecutor(
      testNameBase, tests,
      (testName, tests, features) =>
          executeMathTests(testName, tests, 'webgl', features),
      featuresList);
}

export function describeCustom(
    name: string, tests: Tests, featuresList?: Features[],
    customBeforeEach?: () => void, customAfterEach?: () => void) {
  describeWithFeaturesAndExecutor(
      name, [tests],
      (testName, tests, features) => executeTests(
          testName, tests, features, customBeforeEach, customAfterEach),
      featuresList);
}

type TestExecutor = (testName: string, tests: Tests[], features?: Features) =>
    void;
function describeWithFeaturesAndExecutor(
    testNameBase: string, tests: Tests[], executor: TestExecutor,
    featuresList?: Features[]) {
  if (featuresList != null) {
    featuresList.forEach(features => {
      const testName = testNameBase + ' ' + JSON.stringify(features);
      executor(testName, tests, features);
    });
  } else {
    executor(testNameBase, tests);
  }
}

export function executeMathTests(
    testName: string, tests: Tests[], backendType: BackendType,
    features?: Features) {
  const customBeforeEach = () => {
    Environment.setBackend(backendType);
    ENV.engine.startScope();
  };
  const customAfterEach = () => {
    ENV.engine.endScope(null);
  };

  executeTests(
      testName, tests as Tests[], features, customBeforeEach, customAfterEach);
}

function executeTests(
    testName: string, tests: Tests[], features?: Features,
    customBeforeEach?: () => void, customAfterEach?: () => void) {
  describe(testName, () => {
    beforeEach(() => {
      ENV.setFeatures(features || {});
      ENV.addCustomBackend('webgl', () => new MathBackendWebGL());
      ENV.addCustomBackend('cpu', () => new MathBackendCPU());

      if (customBeforeEach != null) {
        customBeforeEach();
      }
    });

    afterEach(() => {
      if (customAfterEach != null) {
        customAfterEach();
      }
      ENV.reset();
    });

    tests.forEach(test => test());
  });
}

export function assertIsNan(val: number, dtype: DataType) {
  if (!util.isValNaN(val, dtype)) {
    throw new Error(`Value ${val} does not represent NaN for dtype ${dtype}`);
  }
}
