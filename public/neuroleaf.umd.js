(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
    typeof define === 'function' && define.amd ? define(['exports'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.NeuroLeaf = {}));
})(this, (function (exports) { 'use strict';

    // Matrix.ts - Matrix operations for ELM without external dependencies
    class Matrix {
        static multiply(A, B) {
            const result = [];
            for (let i = 0; i < A.length; i++) {
                result[i] = [];
                for (let j = 0; j < B[0].length; j++) {
                    let sum = 0;
                    for (let k = 0; k < B.length; k++) {
                        sum += A[i][k] * B[k][j];
                    }
                    result[i][j] = sum;
                }
            }
            return result;
        }
        static transpose(A) {
            return A[0].map((_, i) => A.map(row => row[i]));
        }
        static identity(size) {
            return Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => (i === j ? 1 : 0)));
        }
        static addRegularization(A, lambda) {
            const result = A.map((row, i) => row.map((val, j) => val + (i === j ? lambda : 0)));
            return result;
        }
        static inverse(A) {
            const n = A.length;
            const I = Matrix.identity(n);
            const M = A.map(row => [...row]);
            for (let i = 0; i < n; i++) {
                let maxEl = Math.abs(M[i][i]);
                let maxRow = i;
                for (let k = i + 1; k < n; k++) {
                    if (Math.abs(M[k][i]) > maxEl) {
                        maxEl = Math.abs(M[k][i]);
                        maxRow = k;
                    }
                }
                [M[i], M[maxRow]] = [M[maxRow], M[i]];
                [I[i], I[maxRow]] = [I[maxRow], I[i]];
                const div = M[i][i];
                if (div === 0)
                    throw new Error("Matrix is singular and cannot be inverted");
                for (let j = 0; j < n; j++) {
                    M[i][j] /= div;
                    I[i][j] /= div;
                }
                for (let k = 0; k < n; k++) {
                    if (k === i)
                        continue;
                    const factor = M[k][i];
                    for (let j = 0; j < n; j++) {
                        M[k][j] -= factor * M[i][j];
                        I[k][j] -= factor * I[i][j];
                    }
                }
            }
            return I;
        }
    }

    // Activations.ts - Common activation functions
    class Activations {
        static relu(x) {
            return Math.max(0, x);
        }
        static leakyRelu(x, alpha = 0.01) {
            return x >= 0 ? x : alpha * x;
        }
        static sigmoid(x) {
            return 1 / (1 + Math.exp(-x));
        }
        static tanh(x) {
            return Math.tanh(x);
        }
        static softmax(arr) {
            const max = Math.max(...arr);
            const exps = arr.map(x => Math.exp(x - max));
            const sum = exps.reduce((a, b) => a + b, 0);
            return exps.map(e => e / sum);
        }
        static apply(matrix, fn) {
            return matrix.map(row => row.map(fn));
        }
        static get(name) {
            switch (name.toLowerCase()) {
                case 'relu': return this.relu;
                case 'leakyrelu': return x => this.leakyRelu(x);
                case 'sigmoid': return this.sigmoid;
                case 'tanh': return this.tanh;
                default: throw new Error(`Unknown activation: ${name}`);
            }
        }
    }

    // ELMConfig.ts - Configuration interface and defaults for ELM-based models
    const defaultConfig = {
        categories: [],
        hiddenUnits: 120,
        maxLen: 15,
        activation: 'relu'
    };

    // Tokenizer.ts - Utility for splitting and tokenizing text inputs
    class Tokenizer {
        constructor(customDelimiter) {
            // Default to splitting on whitespace and punctuation
            this.delimiter = customDelimiter || /[\s,.;!?()\[\]{}"']+/;
        }
        tokenize(text) {
            return text
                .trim()
                .toLowerCase()
                .split(this.delimiter)
                .filter(Boolean); // Remove empty tokens
        }
        ngrams(tokens, n) {
            if (n <= 0 || tokens.length < n)
                return [];
            const result = [];
            for (let i = 0; i <= tokens.length - n; i++) {
                result.push(tokens.slice(i, i + n).join(' '));
            }
            return result;
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

    // ELM.ts - Core ELM logic with TypeScript types
    class ELM {
        constructor(config) {
            var _a, _b;
            const cfg = Object.assign(Object.assign({}, defaultConfig), config);
            this.categories = cfg.categories;
            this.hiddenUnits = cfg.hiddenUnits;
            this.maxLen = cfg.maxLen;
            this.activation = cfg.activation;
            this.charSet = (_a = cfg.charSet) !== null && _a !== void 0 ? _a : 'abcdefghijklmnopqrstuvwxyz';
            this.useTokenizer = (_b = cfg.useTokenizer) !== null && _b !== void 0 ? _b : false;
            this.tokenizerDelimiter = cfg.tokenizerDelimiter;
            this.encoder = new UniversalEncoder({
                charSet: this.charSet,
                maxLen: this.maxLen,
                useTokenizer: this.useTokenizer,
                tokenizerDelimiter: this.tokenizerDelimiter,
                mode: this.useTokenizer ? 'token' : 'char'
            });
            this.model = null;
        }
        oneHot(n, index) {
            return Array.from({ length: n }, (_, i) => (i === index ? 1 : 0));
        }
        pseudoInverse(H, lambda = 1e-3) {
            const Ht = Matrix.transpose(H);
            const HtH = Matrix.multiply(Ht, H);
            const HtH_reg = Matrix.addRegularization(HtH, lambda);
            const HtH_inv = Matrix.inverse(HtH_reg);
            return Matrix.multiply(HtH_inv, Ht);
        }
        randomMatrix(rows, cols) {
            return Array.from({ length: rows }, () => Array.from({ length: cols }, () => Math.random() * 2 - 1));
        }
        setCategories(categories) {
            this.categories = categories;
        }
        train(augmentationOptions) {
            const X = [], Y = [];
            this.categories.forEach((cat, i) => {
                const variants = Augment.generateVariants(cat, this.charSet, augmentationOptions);
                for (const variant of variants) {
                    const vec = this.encoder.normalize(this.encoder.encode(variant));
                    X.push(vec);
                    Y.push(this.oneHot(this.categories.length, i));
                }
            });
            const W = this.randomMatrix(this.hiddenUnits, X[0].length);
            const b = this.randomMatrix(this.hiddenUnits, 1);
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const activationFn = Activations.get(this.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const H_pinv = this.pseudoInverse(H);
            const beta = Matrix.multiply(H_pinv, Y);
            this.model = { W, b, beta };
        }
        predict(text, topK = 5) {
            if (!this.model)
                throw new Error("Model not trained.");
            const vec = this.encoder.normalize(this.encoder.encode(text));
            const { W, b, beta } = this.model;
            const tempH = Matrix.multiply([vec], Matrix.transpose(W));
            const activationFn = Activations.get(this.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const rawOutput = Matrix.multiply(H, beta)[0];
            const probs = Activations.softmax(rawOutput);
            return probs
                .map((p, i) => ({ label: this.categories[i], prob: p }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, topK);
        }
        predictFromVector(inputVec, topK = 5) {
            if (!this.model)
                throw new Error("Model not trained.");
            const { W, b, beta } = this.model;
            const tempH = Matrix.multiply(inputVec, Matrix.transpose(W));
            const activationFn = Activations.get(this.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            return Matrix.multiply(H, beta).map(rawOutput => {
                const probs = Activations.softmax(rawOutput);
                return probs
                    .map((p, i) => ({ label: this.categories[i], prob: p }))
                    .sort((a, b) => b.prob - a.prob)
                    .slice(0, topK);
            });
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

    class FeatureCombinerELM {
        constructor(config) {
            if (typeof config.hiddenUnits !== 'number') {
                throw new Error('FeatureCombinerELM requires hiddenUnits');
            }
            if (!config.activation) {
                throw new Error('FeatureCombinerELM requires activation');
            }
            this.config = Object.assign(Object.assign({}, config), { categories: [], useTokenizer: false // this ELM takes numeric vectors
             });
            this.elm = new ELM(this.config);
        }
        /**
         * Combines encoder vector and metadata into one input vector
         */
        static combineFeatures(encodedVec, meta) {
            return [...encodedVec, ...meta];
        }
        /**
         * Train the ELM using combined features and labels
         */
        train(encoded, metas, labels) {
            const X = encoded.map((vec, i) => FeatureCombinerELM.combineFeatures(vec, metas[i]));
            const categories = [...new Set(labels)];
            this.elm.setCategories(categories);
            const Y = labels.map(label => this.elm.oneHot(categories.length, categories.indexOf(label)));
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
         * Predict from combined input and metadata
         */
        predict(encodedVec, meta, topK = 1) {
            const input = [FeatureCombinerELM.combineFeatures(encodedVec, meta)];
            const [results] = this.elm.predictFromVector(input, topK);
            return results;
        }
    }

    class RefinerELM {
        constructor(config) {
            this.config = Object.assign(Object.assign({}, config), { useTokenizer: false, categories: [] });
            this.elm = new ELM(this.config);
        }
        train(inputs, labels) {
            const categories = [...new Set(labels)];
            this.elm.setCategories(categories);
            const Y = labels.map(label => this.elm.oneHot(categories.length, categories.indexOf(label)));
            const W = this.elm['randomMatrix'](this.config.hiddenUnits, inputs[0].length);
            const b = this.elm['randomMatrix'](this.config.hiddenUnits, 1);
            const tempH = Matrix.multiply(inputs, Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const H_pinv = this.elm['pseudoInverse'](H);
            const beta = Matrix.multiply(H_pinv, Y);
            this.elm['model'] = { W, b, beta };
        }
        predict(vec) {
            const input = [vec];
            const model = this.elm['model'];
            if (!model) {
                throw new Error('EncoderELM model has not been trained yet.');
            }
            const { W, b, beta } = model;
            const tempH = Matrix.multiply(input, Matrix.transpose(W));
            const activationFn = Activations.get(this.config.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const rawOutput = Matrix.multiply(H, beta)[0];
            const probs = Activations.softmax(rawOutput);
            return probs
                .map((p, i) => ({ label: this.elm.categories[i], prob: p }))
                .sort((a, b) => b.prob - a.prob);
        }
    }

    class CharacterLangEncoderELM {
        constructor(config) {
            if (!config.hiddenUnits || !config.activation) {
                throw new Error("CharacterLangEncoderELM requires defined hiddenUnits and activation");
            }
            this.config = Object.assign(Object.assign({}, config), { useTokenizer: true });
            this.elm = new ELM(this.config);
        }
        train(inputStrings, labels) {
            const categories = [...new Set(labels)];
            this.elm.setCategories(categories);
            this.elm.train(); // assumes encoder + categories are set
        }
        /**
         * Returns dense vector (embedding) rather than label prediction
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
            // dense feature vector
            return Matrix.multiply(H, beta)[0];
        }
    }

    exports.Activations = Activations;
    exports.Augment = Augment;
    exports.AutoComplete = AutoComplete;
    exports.CharacterLangEncoderELM = CharacterLangEncoderELM;
    exports.ELM = ELM;
    exports.EncoderELM = EncoderELM;
    exports.FeatureCombinerELM = FeatureCombinerELM;
    exports.IO = IO;
    exports.IntentClassifier = IntentClassifier;
    exports.LanguageClassifier = LanguageClassifier;
    exports.RefinerELM = RefinerELM;
    exports.TextEncoder = TextEncoder;
    exports.Tokenizer = Tokenizer;
    exports.UniversalEncoder = UniversalEncoder;
    exports.bindAutocompleteUI = bindAutocompleteUI;
    exports.defaultConfig = defaultConfig;

}));
//# sourceMappingURL=neuroleaf.umd.js.map
