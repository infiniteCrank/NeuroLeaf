(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
    typeof define === 'function' && define.amd ? define(['exports'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.NeuroLeaf = {}));
})(this, (function (exports) { 'use strict';

    class Matrix {
        constructor(rows, cols, fill = 0) {
            this.rows = rows;
            this.cols = cols;
            this.data = Array.from({ length: rows }, () => Array.from({ length: cols }, () => fill));
        }
        static from2D(values) {
            const m = new Matrix(values.length, values[0].length);
            for (let i = 0; i < m.rows; i++) {
                for (let j = 0; j < m.cols; j++) {
                    m.set(i, j, values[i][j]);
                }
            }
            return m;
        }
        toArray() {
            return this.data.map(row => [...row]);
        }
        get(i, j) {
            return this.data[i][j];
        }
        set(i, j, value) {
            this.data[i][j] = value;
        }
        map(fn) {
            return Matrix.from2D(this.data.map(row => row.map(fn)));
        }
        add(other) {
            this.checkShapeMatch(other);
            return Matrix.from2D(this.data.map((row, i) => row.map((val, j) => val + other.get(i, j))));
        }
        subtract(other) {
            this.checkShapeMatch(other);
            return Matrix.from2D(this.data.map((row, i) => row.map((val, j) => val - other.get(i, j))));
        }
        elementwiseMultiply(other) {
            this.checkShapeMatch(other);
            return Matrix.from2D(this.data.map((row, i) => row.map((val, j) => val * other.get(i, j))));
        }
        multiply(other) {
            if (this.cols !== other.rows) {
                throw new Error('Matrix dimension mismatch for multiplication.');
            }
            const result = new Matrix(this.rows, other.cols);
            for (let i = 0; i < result.rows; i++) {
                for (let j = 0; j < result.cols; j++) {
                    let sum = 0;
                    for (let k = 0; k < this.cols; k++) {
                        sum += this.get(i, k) * other.get(k, j);
                    }
                    result.set(i, j, sum);
                }
            }
            return result;
        }
        dot(other) {
            return this.multiply(other);
        }
        transpose() {
            const result = new Matrix(this.cols, this.rows);
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    result.set(j, i, this.get(i, j));
                }
            }
            return result;
        }
        inverse() {
            if (this.rows !== this.cols) {
                throw new Error("Only square matrices can be inverted.");
            }
            const n = this.rows;
            const A = this.toArray();
            const I = Matrix.identity(n).toArray();
            for (let i = 0; i < n; i++) {
                let factor = A[i][i];
                if (factor === 0) {
                    let swapRow = A.findIndex((r, k) => k > i && r[i] !== 0);
                    if (swapRow === -1)
                        throw new Error("Matrix is singular.");
                    [A[i], A[swapRow]] = [A[swapRow], A[i]];
                    [I[i], I[swapRow]] = [I[swapRow], I[i]];
                    factor = A[i][i];
                }
                for (let j = 0; j < n; j++) {
                    A[i][j] /= factor;
                    I[i][j] /= factor;
                }
                for (let k = 0; k < n; k++) {
                    if (k !== i) {
                        const scale = A[k][i];
                        for (let j = 0; j < n; j++) {
                            A[k][j] -= scale * A[i][j];
                            I[k][j] -= scale * I[i][j];
                        }
                    }
                }
            }
            return Matrix.from2D(I);
        }
        static identity(n) {
            const m = new Matrix(n, n);
            for (let i = 0; i < n; i++) {
                m.set(i, i, 1);
            }
            return m;
        }
        static random(rows, cols, min = -1, max = 1) {
            const m = new Matrix(rows, cols);
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    m.set(i, j, Math.random() * (max - min) + min);
                }
            }
            return m;
        }
        checkShapeMatch(other) {
            if (this.rows !== other.rows || this.cols !== other.cols) {
                throw new Error("Matrix shape mismatch.");
            }
        }
        reshape(newRows, newCols) {
            if (newRows * newCols !== this.rows * this.cols) {
                throw new Error("New shape must match total number of elements.");
            }
            const flat = this.data.flat();
            const reshaped = [];
            for (let i = 0; i < newRows; i++) {
                reshaped.push(flat.slice(i * newCols, (i + 1) * newCols));
            }
            return Matrix.from2D(reshaped);
        }
        sum(axis) {
            if (axis === 0) {
                // column-wise sum
                return Array.from({ length: this.cols }, (_, j) => this.data.reduce((sum, row) => sum + row[j], 0));
            }
            else if (axis === 1) {
                // row-wise sum
                return this.data.map(row => row.reduce((a, b) => a + b, 0));
            }
            else {
                // full sum
                return this.data.flat().reduce((a, b) => a + b, 0);
            }
        }
        mean(axis) {
            const sumResult = this.sum(axis);
            if (axis === 0 && Array.isArray(sumResult)) {
                return sumResult.map(sum => sum / this.rows);
            }
            else if (axis === 1 && Array.isArray(sumResult)) {
                return sumResult.map(sum => sum / this.cols);
            }
            else if (typeof sumResult === 'number') {
                return sumResult / (this.rows * this.cols);
            }
            else {
                throw new Error("Unexpected shape in mean()");
            }
        }
        flatten() {
            return this.data.flat();
        }
        argmax() {
            let maxVal = -Infinity;
            let maxRow = 0;
            let maxCol = 0;
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    const val = this.get(i, j);
                    if (val > maxVal) {
                        maxVal = val;
                        maxRow = i;
                        maxCol = j;
                    }
                }
            }
            return { row: maxRow, col: maxCol, value: maxVal };
        }
        softmax() {
            const flat = this.flatten();
            const max = Math.max(...flat); // for numerical stability
            const exps = flat.map(x => Math.exp(x - max));
            const sum = exps.reduce((a, b) => a + b, 0);
            const softmaxValues = exps.map(x => x / sum);
            // Return as 1-row matrix
            return Matrix.from2D([softmaxValues]);
        }
        clip(min, max) {
            return this.map(x => Math.max(min, Math.min(max, x)));
        }
        normalize() {
            const flat = this.flatten();
            const norm = Math.sqrt(flat.reduce((sum, x) => sum + x * x, 0)) || 1;
            return this.map(x => x / norm);
        }
        standardize() {
            const flat = this.flatten();
            const mean = flat.reduce((sum, x) => sum + x, 0) / flat.length;
            const variance = flat.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / flat.length;
            const std = Math.sqrt(variance) || 1;
            return this.map(x => (x - mean) / std);
        }
        minMaxScale() {
            const flat = this.flatten();
            const min = Math.min(...flat);
            const max = Math.max(...flat);
            const range = max - min || 1;
            return this.map(x => (x - min) / range);
        }
        rowWiseNormalize() {
            const newRows = this.toArray().map(row => {
                const norm = Math.sqrt(row.reduce((sum, x) => sum + x * x, 0)) || 1;
                return row.map(x => x / norm);
            });
            return Matrix.from2D(newRows);
        }
        zscore(axis) {
            const rows = this.rows;
            const cols = this.cols;
            if (axis === 0) {
                const means = this.mean(0);
                const stds = this.toArray()[0].map((_, j) => {
                    const col = this.data.map(row => row[j]);
                    const mean = means[j];
                    const variance = col.reduce((sum, x) => sum + Math.pow((x - mean), 2), 0) / rows;
                    return Math.sqrt(variance) || 1;
                });
                const z = this.toArray().map(row => row.map((val, j) => (val - means[j]) / stds[j]));
                return Matrix.from2D(z);
            }
            else if (axis === 1) {
                const z = this.toArray().map(row => {
                    const mean = row.reduce((a, b) => a + b, 0) / cols;
                    const std = Math.sqrt(row.reduce((a, b) => a + Math.pow((b - mean), 2), 0) / cols) || 1;
                    return row.map(x => (x - mean) / std);
                });
                return Matrix.from2D(z);
            }
            else {
                return this.standardize();
            }
        }
        describe(axis = 0) {
            const data = this.toArray();
            const dim = axis === 0 ? this.cols : this.rows;
            const out = {
                min: [],
                max: [],
                mean: [],
                std: [],
                median: []
            };
            for (let i = 0; i < dim; i++) {
                const values = axis === 0 ? data.map(row => row[i]) : data[i];
                const sorted = [...values].sort((a, b) => a - b);
                const mid = Math.floor(sorted.length / 2);
                const median = sorted.length % 2 === 0
                    ? (sorted[mid - 1] + sorted[mid]) / 2
                    : sorted[mid];
                const mean = values.reduce((a, b) => a + b, 0) / values.length;
                const std = Math.sqrt(values.reduce((sum, x) => sum + Math.pow((x - mean), 2), 0) / values.length);
                out.min.push(Math.min(...values));
                out.max.push(Math.max(...values));
                out.mean.push(mean);
                out.std.push(std);
                out.median.push(median);
            }
            // Export to console for dev inspection
            if (typeof window !== 'undefined') {
                console.log('[Matrix.describe]', Object.assign({ axis }, out));
            }
            return out;
        }
        static inverse(m) {
            return m.inverse();
        }
        static transpose(m) {
            return m.transpose();
        }
        static addRegularization(matrix, lambda) {
            return matrix.map((row, i) => row.map((val, j) => (i === j ? val + lambda : val)));
        }
        static addRegularizationMatrix(m, lambda) {
            const updated = m.toArray().map((row, i) => row.map((val, j) => (i === j ? val + lambda : val)));
            return Matrix.from2D(updated);
        }
        static mean(m, axis) {
            return m.mean(axis);
        }
        static map(m, fn) {
            return m.map(fn);
        }
        static multiply(a, b) {
            return a.multiply(b);
        }
    }

    class Tokenizer {
        constructor(config) {
            var _a, _b, _c;
            this.charSet = config.charSet.split('');
            this.charToIndex = Object.fromEntries(this.charSet.map((c, i) => [c, i]));
            this.maxLen = (_a = config.maxLen) !== null && _a !== void 0 ? _a : 30;
            this.useTokenizer = (_b = config.useTokenizer) !== null && _b !== void 0 ? _b : false;
            this.tokenizerDelimiter = (_c = config.tokenizerDelimiter) !== null && _c !== void 0 ? _c : /\s+/;
        }
        encode(input) {
            const vec = new Array(this.charSet.length * this.maxLen).fill(0);
            const tokens = this.useTokenizer
                ? input.split(this.tokenizerDelimiter)
                : input.split('');
            for (let i = 0; i < Math.min(tokens.length, this.maxLen); i++) {
                const token = tokens[i];
                const index = this.charToIndex[token];
                if (index !== undefined) {
                    vec[i * this.charSet.length + index] = 1;
                }
            }
            return vec;
        }
    }

    const Activations = {
        relu: (x) => Math.max(0, x),
        sigmoid: (x) => 1 / (1 + Math.exp(-x)),
        tanh: (x) => Math.tanh(x),
        reluMatrix: (m) => m.map((x) => Math.max(0, x)),
        sigmoidMatrix: (m) => m.map((x) => 1 / (1 + Math.exp(-x))),
        tanhMatrix: (m) => m.map((x) => Math.tanh(x))
    };

    function normalize(vec) {
        const norm = Math.sqrt(vec.reduce((acc, v) => acc + v * v, 0)) || 1;
        return vec.map(v => v / norm);
    }
    class ELM {
        constructor(config) {
            this.categories = [];
            this.config = config;
            this.encoder = new Tokenizer({
                charSet: config.charSet,
                maxLen: config.maxLen,
                useTokenizer: config.useTokenizer,
                tokenizerDelimiter: config.tokenizerDelimiter
            });
            this.activator = Activations[`${this.config.activation}Matrix`];
        }
        setCategories(categories) {
            this.categories = categories;
        }
        oneHot(n, index) {
            return Array.from({ length: n }, (_, i) => (i === index ? 1 : 0));
        }
        train(trainingData) {
            this.categories = this.config.categories || Array.from(new Set(trainingData.map(d => d.label)));
            const labelToIndex = Object.fromEntries(this.categories.map((label, i) => [label, i]));
            const X = trainingData.map(d => normalize(this.encoder.encode(d.text)));
            const Y = trainingData.map(d => {
                const row = Array(this.categories.length).fill(0);
                row[labelToIndex[d.label]] = 1;
                return row;
            });
            const Xmat = Matrix.from2D(X).transpose();
            const H = this.activator(Matrix.random(this.config.hiddenUnits, Xmat.rows).dot(Xmat));
            this.hiddenMatrix = H;
            const Ymat = Matrix.from2D(Y).transpose();
            const beta = Matrix.inverse(H.transpose().dot(H)).dot(H.transpose()).dot(Ymat);
            this.beta = beta;
        }
        predict(input) {
            const x = normalize(this.encoder.encode(input));
            const h = this.activator(this.hiddenMatrix.dot(Matrix.from2D([x]).transpose()));
            const out = this.beta.transpose().dot(h).toArray().map(r => r[0]);
            const temp = this.config.temperature || 1;
            const exps = out.map(v => Math.exp(v / temp));
            const sum = exps.reduce((a, b) => a + b, 0);
            const softmax = exps.map(v => v / sum);
            return this.categories.map((label, i) => ({ label, prob: softmax[i] }))
                .sort((a, b) => b.prob - a.prob);
        }
    }

    // BindUI.ts - Utility to bind ELM model to HTML inputs and outputs
    function bindAutocompleteUI({ model, inputElement, outputElement, topK = 5 }) {
        inputElement.addEventListener('input', () => {
            const typed = inputElement.value.trim();
            if (typed.length === 0) {
                outputElement.innerHTML = '<em>Start typing...</em>';
                return;
            }
            try {
                const results = model.predict(typed, topK);
                outputElement.innerHTML = results.map(r => `
                <div><strong>${r.label}</strong>: ${(r.prob * 100).toFixed(1)}%</div>
            `).join('');
            }
            catch (e) {
                const message = e instanceof Error ? e.message : 'Unknown error';
                outputElement.innerHTML = `<span style="color: red;">Error: ${message}</span>`;
            }
        });
    }

    // Presets.ts - Reusable configuration presets for ELM
    const EnglishTokenPreset = {
        categories: [],
        hiddenUnits: 120,
        maxLen: 20,
        activation: 'relu',
        charSet: 'abcdefghijklmnopqrstuvwxyz',
        useTokenizer: true,
        tokenizerDelimiter: /[\s,.;!?()\[\]{}"']+/
    };

    // AutoComplete.ts - High-level autocomplete controller using ELM
    class AutoComplete {
        constructor(categories, options) {
            this.model = new ELM(Object.assign(Object.assign({}, EnglishTokenPreset), { categories }));
            // Train the model, safely handling optional augmentationOptions
            this.model.train(options === null || options === void 0 ? void 0 : options.augmentationOptions);
            bindAutocompleteUI({
                model: this.model,
                inputElement: options.inputElement,
                outputElement: options.outputElement,
                topK: options.topK
            });
        }
        predict(input, topN = 1) {
            return this.model.predict(input).slice(0, topN).map(p => ({
                completion: p.label,
                prob: p.prob
            }));
        }
        getModel() {
            return this.model;
        }
    }

    /**
     * EncoderELM: Uses an ELM to convert strings into dense feature vectors.
     */
    class EncoderELM {
        constructor(config) {
            if (typeof config.hiddenUnits !== 'number') {
                throw new Error('EncoderELM requires config.hiddenUnits to be defined as a number');
            }
            if (!config.activation) {
                throw new Error('EncoderELM requires config.activation to be defined');
            }
            this.config = Object.assign(Object.assign({}, config), { categories: [], useTokenizer: true });
            this.elm = new ELM(this.config);
        }
        /**
         * Custom training method for string → vector encoding.
         */
        train(inputStrings, targetVectors) {
            const X = inputStrings.map(s => this.elm.encoder.normalize(this.elm.encoder.encode(s)));
            const Y = targetVectors;
            const hiddenUnits = this.config.hiddenUnits;
            const inputDim = X[0].length;
            const W = this.elm['randomMatrix'](hiddenUnits, inputDim);
            const b = this.elm['randomMatrix'](hiddenUnits, 1);
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const H_pinv = this.elm['pseudoInverse'](H);
            const beta = Matrix.multiply(H_pinv, Y);
            this.elm['model'] = { W, b, beta };
        }
        /**
         * Encodes an input string into a dense feature vector using the trained model.
         */
        encode(text) {
            const vec = this.elm.encoder.normalize(this.elm.encoder.encode(text));
            const model = this.elm['model'];
            if (!model) {
                throw new Error('EncoderELM model has not been trained yet.');
            }
            const { W, b, beta } = model;
            const tempH = Matrix.multiply([vec], Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            return Matrix.multiply(H, beta)[0];
        }
    }

    // intentClassifier.ts - ELM-based intent classification engine
    class IntentClassifier {
        constructor(config) {
            this.model = new ELM(config);
        }
        train(textLabelPairs, augmentationOptions) {
            const labelSet = Array.from(new Set(textLabelPairs.map(p => p.label)));
            Object.assign(Object.assign({}, this.model), { categories: labelSet });
            this.model.train(augmentationOptions);
        }
        predict(text, topK = 1, threshold = 0) {
            return this.model.predict(text, topK).filter(r => r.prob >= threshold);
        }
        predictBatch(texts, topK = 1, threshold = 0) {
            return texts.map(text => this.predict(text, topK, threshold));
        }
        oneHot(n, index) {
            return Array.from({ length: n }, (_, i) => (i === index ? 1 : 0));
        }
    }

    // IO.ts - Import/export utilities for labeled training data
    class IO {
        static importJSON(json) {
            try {
                const data = JSON.parse(json);
                if (!Array.isArray(data))
                    throw new Error('Invalid format');
                return data.filter(item => typeof item.text === 'string' && typeof item.label === 'string');
            }
            catch (err) {
                console.error('Failed to parse training data JSON:', err);
                return [];
            }
        }
        static exportJSON(pairs) {
            return JSON.stringify(pairs, null, 2);
        }
        static importDelimited(text, delimiter = ',', hasHeader = true) {
            var _a, _b, _c, _d;
            const lines = text.trim().split('\n');
            const examples = [];
            const headers = hasHeader
                ? lines[0].split(delimiter).map(h => h.trim().toLowerCase())
                : lines[0].split(delimiter).length === 1
                    ? ['label']
                    : ['text', 'label'];
            const startIndex = hasHeader ? 1 : 0;
            for (let i = startIndex; i < lines.length; i++) {
                const parts = lines[i].split(delimiter);
                if (parts.length === 1) {
                    examples.push({ text: parts[0].trim(), label: parts[0].trim() });
                }
                else {
                    const textIdx = headers.indexOf('text');
                    const labelIdx = headers.indexOf('label');
                    const text = textIdx !== -1 ? (_a = parts[textIdx]) === null || _a === void 0 ? void 0 : _a.trim() : (_b = parts[0]) === null || _b === void 0 ? void 0 : _b.trim();
                    const label = labelIdx !== -1 ? (_c = parts[labelIdx]) === null || _c === void 0 ? void 0 : _c.trim() : (_d = parts[1]) === null || _d === void 0 ? void 0 : _d.trim();
                    if (text && label) {
                        examples.push({ text, label });
                    }
                }
            }
            return examples;
        }
        static exportDelimited(pairs, delimiter = ',', includeHeader = true) {
            const header = includeHeader ? `text${delimiter}label\n` : '';
            const rows = pairs.map(p => `${p.text.replace(new RegExp(delimiter, 'g'), '')}${delimiter}${p.label.replace(new RegExp(delimiter, 'g'), '')}`);
            return header + rows.join('\n');
        }
        static importCSV(csv, hasHeader = true) {
            return this.importDelimited(csv, ',', hasHeader);
        }
        static exportCSV(pairs, includeHeader = true) {
            return this.exportDelimited(pairs, ',', includeHeader);
        }
        static importTSV(tsv, hasHeader = true) {
            return this.importDelimited(tsv, '\t', hasHeader);
        }
        static exportTSV(pairs, includeHeader = true) {
            return this.exportDelimited(pairs, '\t', includeHeader);
        }
        static inferSchemaFromCSV(csv) {
            var _a;
            const lines = csv.trim().split('\n');
            if (lines.length === 0)
                return { fields: [] };
            const header = lines[0].split(',').map(h => h.trim().toLowerCase());
            const row = ((_a = lines[1]) === null || _a === void 0 ? void 0 : _a.split(',')) || [];
            const fields = header.map((name, i) => {
                var _a;
                const sample = (_a = row[i]) === null || _a === void 0 ? void 0 : _a.trim();
                let type = 'unknown';
                if (!sample)
                    type = 'unknown';
                else if (!isNaN(Number(sample)))
                    type = 'number';
                else if (sample === 'true' || sample === 'false')
                    type = 'boolean';
                else
                    type = 'string';
                return { name, type };
            });
            const suggestedMapping = {
                text: header.find(h => h.includes('text') || h.includes('utterance') || h.includes('input')) || header[0],
                label: header.find(h => h.includes('label') || h.includes('intent') || h.includes('tag')) || header[1] || header[0],
            };
            return { fields, suggestedMapping };
        }
        static inferSchemaFromJSON(json) {
            try {
                const data = JSON.parse(json);
                if (!Array.isArray(data) || data.length === 0 || typeof data[0] !== 'object')
                    return { fields: [] };
                const keys = Object.keys(data[0]);
                const fields = keys.map(key => {
                    const val = data[0][key];
                    let type = 'unknown';
                    if (typeof val === 'string')
                        type = 'string';
                    else if (typeof val === 'number')
                        type = 'number';
                    else if (typeof val === 'boolean')
                        type = 'boolean';
                    return { name: key.toLowerCase(), type };
                });
                const suggestedMapping = {
                    text: keys.find(k => k.toLowerCase().includes('text') || k.toLowerCase().includes('utterance') || k.toLowerCase().includes('input')) || keys[0],
                    label: keys.find(k => k.toLowerCase().includes('label') || k.toLowerCase().includes('intent') || k.toLowerCase().includes('tag')) || keys[1] || keys[0],
                };
                return { fields, suggestedMapping };
            }
            catch (err) {
                console.error('Failed to infer schema from JSON:', err);
                return { fields: [] };
            }
        }
    }

    class LanguageClassifier {
        constructor(config) {
            this.trainSamples = {};
            this.config = config;
            this.elm = new ELM(config);
        }
        loadTrainingData(raw, format = 'json') {
            switch (format) {
                case 'csv':
                    return IO.importCSV(raw);
                case 'tsv':
                    return IO.importTSV(raw);
                case 'json':
                default:
                    return IO.importJSON(raw);
            }
        }
        train(data) {
            const categories = [...new Set(data.map(d => d.label))];
            this.elm.setCategories(categories);
            data.forEach(({ text, label }) => {
                if (!this.trainSamples[label])
                    this.trainSamples[label] = [];
                this.trainSamples[label].push(text);
            });
            this.elm.train();
        }
        predict(text, topK = 3) {
            return this.elm.predict(text, topK);
        }
        /**
         * Train the classifier using already-encoded vectors.
         * Each vector must be paired with its label.
         */
        trainVectors(data) {
            const categories = [...new Set(data.map(d => d.label))];
            this.elm.setCategories(categories);
            const X = data.map(d => d.vector);
            const Y = data.map(d => this.elm.oneHot(categories.length, categories.indexOf(d.label)));
            const W = this.elm['randomMatrix'](this.config.hiddenUnits, X[0].length);
            const b = this.elm['randomMatrix'](this.config.hiddenUnits, 1);
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const H_pinv = this.elm['pseudoInverse'](H);
            const beta = Matrix.multiply(H_pinv, Y);
            this.elm['model'] = { W, b, beta };
        }
        /**
         * Predict language directly from a dense vector representation.
         */
        predictFromVector(vec, topK = 1) {
            const model = this.elm['model'];
            if (!model) {
                throw new Error('EncoderELM model has not been trained yet.');
            }
            const { W, b, beta } = model;
            const tempH = Matrix.multiply([vec], Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const rawOutput = Matrix.multiply(H, beta)[0];
            const probs = Activations.softmax(rawOutput);
            return probs
                .map((p, i) => ({ label: this.elm.categories[i], prob: p }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, topK);
        }
    }

    // TextEncoder.ts - Text preprocessing and one-hot encoding for ELM
    const defaultTextEncoderConfig = {
        charSet: 'abcdefghijklmnopqrstuvwxyz',
        maxLen: 15,
        useTokenizer: false
    };
    class TextEncoder {
        constructor(config = {}) {
            const cfg = Object.assign(Object.assign({}, defaultTextEncoderConfig), config);
            this.charSet = cfg.charSet;
            this.charSize = cfg.charSet.length;
            this.maxLen = cfg.maxLen;
            this.useTokenizer = cfg.useTokenizer;
            if (this.useTokenizer) {
                this.tokenizer = new Tokenizer(config.tokenizerDelimiter);
            }
        }
        charToOneHot(c) {
            const index = this.charSet.indexOf(c.toLowerCase());
            const vec = Array(this.charSize).fill(0);
            if (index !== -1)
                vec[index] = 1;
            return vec;
        }
        textToVector(text) {
            let cleaned;
            if (this.useTokenizer && this.tokenizer) {
                const tokens = this.tokenizer.tokenize(text).join('');
                cleaned = tokens.slice(0, this.maxLen).padEnd(this.maxLen, ' ');
            }
            else {
                cleaned = text.toLowerCase().replace(new RegExp(`[^${this.charSet}]`, 'g'), '').padEnd(this.maxLen, ' ').slice(0, this.maxLen);
            }
            const vec = [];
            for (let i = 0; i < cleaned.length; i++) {
                vec.push(...this.charToOneHot(cleaned[i]));
            }
            return vec;
        }
        normalizeVector(v) {
            const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
            return norm > 0 ? v.map(x => x / norm) : v;
        }
        getVectorSize() {
            return this.charSize * this.maxLen;
        }
        getCharSet() {
            return this.charSet;
        }
        getMaxLen() {
            return this.maxLen;
        }
    }

    // UniversalEncoder.ts - Automatically selects appropriate encoder (char or token based)
    const defaultUniversalConfig = {
        charSet: 'abcdefghijklmnopqrstuvwxyz',
        maxLen: 15,
        useTokenizer: false,
        mode: 'char'
    };
    class UniversalEncoder {
        constructor(config = {}) {
            const merged = Object.assign(Object.assign({}, defaultUniversalConfig), config);
            const useTokenizer = merged.mode === 'token';
            this.encoder = new TextEncoder({
                charSet: merged.charSet,
                maxLen: merged.maxLen,
                useTokenizer,
                tokenizerDelimiter: config.tokenizerDelimiter
            });
        }
        encode(text) {
            return this.encoder.textToVector(text);
        }
        normalize(v) {
            return this.encoder.normalizeVector(v);
        }
        getVectorSize() {
            return this.encoder.getVectorSize();
        }
    }

    // ELMConfig.ts - Configuration interface and defaults for ELM-based models
    const defaultConfig = {
        categories: [],
        hiddenUnits: 120,
        maxLen: 15,
        activation: 'relu'
    };

    // Augment.ts - Basic augmentation utilities for category training examples
    class Augment {
        static addSuffix(text, suffixes) {
            return suffixes.map(suffix => `${text} ${suffix}`);
        }
        static addPrefix(text, prefixes) {
            return prefixes.map(prefix => `${prefix} ${text}`);
        }
        static addNoise(text, charSet, noiseRate = 0.1) {
            const chars = text.split('');
            for (let i = 0; i < chars.length; i++) {
                if (Math.random() < noiseRate) {
                    const randomChar = charSet[Math.floor(Math.random() * charSet.length)];
                    chars[i] = randomChar;
                }
            }
            return chars.join('');
        }
        static mix(text, mixins) {
            return mixins.map(m => `${text} ${m}`);
        }
        static generateVariants(text, charSet, options) {
            const variants = [text];
            if (options === null || options === void 0 ? void 0 : options.suffixes) {
                variants.push(...this.addSuffix(text, options.suffixes));
            }
            if (options === null || options === void 0 ? void 0 : options.prefixes) {
                variants.push(...this.addPrefix(text, options.prefixes));
            }
            if (options === null || options === void 0 ? void 0 : options.includeNoise) {
                variants.push(this.addNoise(text, charSet));
            }
            return variants;
        }
    }

    exports.Activations = Activations;
    exports.Augment = Augment;
    exports.AutoComplete = AutoComplete;
    exports.ELM = ELM;
    exports.EncoderELM = EncoderELM;
    exports.IO = IO;
    exports.IntentClassifier = IntentClassifier;
    exports.LanguageClassifier = LanguageClassifier;
    exports.TextEncoder = TextEncoder;
    exports.Tokenizer = Tokenizer;
    exports.UniversalEncoder = UniversalEncoder;
    exports.bindAutocompleteUI = bindAutocompleteUI;
    exports.defaultConfig = defaultConfig;

}));
//# sourceMappingURL=neuroleaf.umd.js.map
