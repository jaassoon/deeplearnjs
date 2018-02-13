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
export interface Docs { headings: DocHeading[]; }

export interface DocHeading {
  name: string;
  description?: string;
  subheadings: DocSubheading[];
}

export interface DocSubheading {
  name: string;
  description?: string;
  symbols?: DocSymbol[];
  // Only used at initialization for sort-order. Pins by displayName, not symbol
  // name (so that we use namespaces).
  pin?: string[];
}

export type DocSymbol = DocFunction|DocClass;

export interface DocClass {
  symbolName: string;
  displayName: string;
  documentation: string;
  fileName: string;
  githubUrl: string;
  methods: DocFunction[];

  isClass: true;
}

export interface DocFunction {
  symbolName: string;
  displayName: string;
  documentation: string;
  fileName: string;
  githubUrl: string;
  parameters: DocFunctionParam[];

  paramStr: string;
  returnType: string;

  isFunction: true;
}

export interface DocFunctionParam {
  name: string;
  type: string;
  optional: boolean;
  documentation: string;
}
