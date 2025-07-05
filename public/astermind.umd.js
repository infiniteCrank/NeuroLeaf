(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
    typeof define === 'function' && define.amd ? define(['exports'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.astermind = {}));
})(this, (function (exports) { 'use strict';

    class Matrix {
        constructor(data) {
            this.data = data;
        }
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
            return A.map((row, i) => row.map((val, j) => val + (i === j ? lambda : 0)));
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
        static random(rows, cols, min, max) {
            const data = [];
            for (let i = 0; i < rows; i++) {
                const row = [];
                for (let j = 0; j < cols; j++) {
                    row.push(Math.random() * (max - min) + min);
                }
                data.push(row);
            }
            return new Matrix(data);
        }
        static fromArray(array) {
            return new Matrix(array);
        }
        toArray() {
            return this.data;
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
        hiddenUnits: 50,
        maxLen: 30,
        weightInit: "uniform",
        activation: 'relu',
        charSet: 'abcdefghijklmnopqrstuvwxyz',
        useTokenizer: false,
        tokenizerDelimiter: /\s+/,
    };

    class Tokenizer {
        constructor(customDelimiter) {
            this.delimiter = customDelimiter || /[\s,.;!?()\[\]{}"']+/;
        }
        tokenize(text) {
            if (typeof text !== 'string') {
                console.warn('[Tokenizer] Expected a string, got:', typeof text, text);
                try {
                    text = String(text !== null && text !== void 0 ? text : '');
                }
                catch (_a) {
                    return [];
                }
            }
            return text
                .trim()
                .toLowerCase()
                .split(this.delimiter)
                .filter(Boolean);
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
            var _a, _b, _c, _d, _e, _f, _g, _h, _j;
            const cfg = Object.assign(Object.assign({}, defaultConfig), config);
            this.categories = cfg.categories;
            this.hiddenUnits = cfg.hiddenUnits;
            this.maxLen = cfg.maxLen;
            this.activation = cfg.activation;
            this.charSet = (_a = cfg.charSet) !== null && _a !== void 0 ? _a : 'abcdefghijklmnopqrstuvwxyz';
            this.useTokenizer = (_b = cfg.useTokenizer) !== null && _b !== void 0 ? _b : false;
            this.tokenizerDelimiter = cfg.tokenizerDelimiter;
            this.config = cfg;
            this.metrics = this.config.metrics;
            this.verbose = (_d = (_c = cfg.log) === null || _c === void 0 ? void 0 : _c.verbose) !== null && _d !== void 0 ? _d : true;
            this.modelName = (_f = (_e = cfg.log) === null || _e === void 0 ? void 0 : _e.modelName) !== null && _f !== void 0 ? _f : 'Unnamed ELM Model';
            this.logToFile = (_h = (_g = cfg.log) === null || _g === void 0 ? void 0 : _g.toFile) !== null && _h !== void 0 ? _h : false;
            this.dropout = (_j = cfg.dropout) !== null && _j !== void 0 ? _j : 0;
            this.encoder = new UniversalEncoder({
                charSet: this.charSet,
                maxLen: this.maxLen,
                useTokenizer: this.useTokenizer,
                tokenizerDelimiter: this.tokenizerDelimiter,
                mode: this.useTokenizer ? 'token' : 'char'
            });
            this.inputWeights = Matrix.fromArray(this.randomMatrix(cfg.hiddenUnits, cfg.maxLen));
            this.biases = Matrix.fromArray(this.randomMatrix(cfg.hiddenUnits, 1));
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
            if (this.config.weightInit === "xavier") {
                if (this.verbose)
                    console.log(`✨ Xavier init with limit sqrt(6/${rows}+${cols})`);
                const limit = Math.sqrt(6 / (rows + cols));
                return Array.from({ length: rows }, () => Array.from({ length: cols }, () => Math.random() * 2 * limit - limit));
            }
            else {
                if (this.verbose)
                    console.log(`✨ Uniform init [-1,1]`);
                return Array.from({ length: rows }, () => Array.from({ length: cols }, () => Math.random() * 2 - 1));
            }
        }
        setCategories(categories) {
            this.categories = categories;
        }
        loadModelFromJSON(json) {
            try {
                const parsed = JSON.parse(json);
                this.model = parsed;
                this.savedModelJSON = json;
                if (this.verbose)
                    console.log(`✅ ${this.modelName} Model loaded from JSON`);
            }
            catch (e) {
                console.error(`❌ Failed to load ${this.modelName} model from JSON:`, e);
            }
        }
        trainFromData(X, Y, options) {
            const reuseWeights = (options === null || options === void 0 ? void 0 : options.reuseWeights) === true;
            let W, b;
            if (reuseWeights && this.model) {
                W = this.model.W;
                b = this.model.b;
                if (this.verbose)
                    console.log("🔄 Reusing existing weights/biases for training.");
            }
            else {
                W = this.randomMatrix(this.hiddenUnits, X[0].length);
                b = this.randomMatrix(this.hiddenUnits, 1);
                if (this.verbose)
                    console.log("✨ Initializing fresh weights/biases for training.");
            }
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const activationFn = Activations.get(this.activation);
            let H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            if (this.dropout > 0) {
                const keepProb = 1 - this.dropout;
                for (let i = 0; i < H.length; i++) {
                    for (let j = 0; j < H[0].length; j++) {
                        if (Math.random() < this.dropout) {
                            H[i][j] = 0;
                        }
                        else {
                            H[i][j] /= keepProb;
                        }
                    }
                }
            }
            if (options === null || options === void 0 ? void 0 : options.weights) {
                const W_arr = options.weights;
                if (W_arr.length !== H.length) {
                    throw new Error(`Weight array length ${W_arr.length} does not match sample count ${H.length}`);
                }
                // Scale each row by sqrt(weight)
                H = H.map((row, i) => row.map(x => x * Math.sqrt(W_arr[i])));
                Y = Y.map((row, i) => row.map(x => x * Math.sqrt(W_arr[i])));
            }
            const H_pinv = this.pseudoInverse(H);
            const beta = Matrix.multiply(H_pinv, Y);
            this.model = { W, b, beta };
            const predictions = Matrix.multiply(H, beta);
            if (this.metrics) {
                const rmse = this.calculateRMSE(Y, predictions);
                const mae = this.calculateMAE(Y, predictions);
                const acc = this.calculateAccuracy(Y, predictions);
                const f1 = this.calculateF1Score(Y, predictions);
                const ce = this.calculateCrossEntropy(Y, predictions);
                const r2 = this.calculateR2Score(Y, predictions);
                const results = {};
                let allPassed = true;
                if (this.metrics.rmse !== undefined) {
                    results.rmse = rmse;
                    if (rmse > this.metrics.rmse)
                        allPassed = false;
                }
                if (this.metrics.mae !== undefined) {
                    results.mae = mae;
                    if (mae > this.metrics.mae)
                        allPassed = false;
                }
                if (this.metrics.accuracy !== undefined) {
                    results.accuracy = acc;
                    if (acc < this.metrics.accuracy)
                        allPassed = false;
                }
                if (this.metrics.f1 !== undefined) {
                    results.f1 = f1;
                    if (f1 < this.metrics.f1)
                        allPassed = false;
                }
                if (this.metrics.crossEntropy !== undefined) {
                    results.crossEntropy = ce;
                    if (ce > this.metrics.crossEntropy)
                        allPassed = false;
                }
                if (this.metrics.r2 !== undefined) {
                    results.r2 = r2;
                    if (r2 < this.metrics.r2)
                        allPassed = false;
                }
                if (this.verbose)
                    this.logMetrics(results);
                if (allPassed) {
                    this.savedModelJSON = JSON.stringify(this.model);
                    if (this.verbose)
                        console.log("✅ Model passed thresholds and was saved to JSON.");
                    if (this.config.exportFileName) {
                        this.saveModelAsJSONFile(this.config.exportFileName);
                    }
                }
                else {
                    if (this.verbose)
                        console.log("❌ Model not saved: One or more thresholds not met.");
                }
            }
            else {
                // No metrics—always save the model
                this.savedModelJSON = JSON.stringify(this.model);
                if (this.verbose)
                    console.log("✅ Model trained with no metrics—saved by default.");
                if (this.config.exportFileName) {
                    this.saveModelAsJSONFile(this.config.exportFileName);
                }
            }
        }
        train(augmentationOptions, weights) {
            const X = [];
            let Y = [];
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
            let H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            if (this.dropout > 0) {
                const keepProb = 1 - this.dropout;
                for (let i = 0; i < H.length; i++) {
                    for (let j = 0; j < H[0].length; j++) {
                        if (Math.random() < this.dropout) {
                            H[i][j] = 0;
                        }
                        else {
                            H[i][j] /= keepProb;
                        }
                    }
                }
            }
            if (weights) {
                if (weights.length !== H.length) {
                    throw new Error(`Weight array length ${weights.length} does not match sample count ${H.length}`);
                }
                // Scale each row of H and Y by sqrt(weight)
                H = H.map((row, i) => row.map(x => x * Math.sqrt(weights[i])));
                Y = Y.map((row, i) => row.map(x => x * Math.sqrt(weights[i])));
            }
            const H_pinv = this.pseudoInverse(H);
            const beta = Matrix.multiply(H_pinv, Y);
            this.model = { W, b, beta };
            const predictions = Matrix.multiply(H, beta);
            if (this.metrics) {
                const rmse = this.calculateRMSE(Y, predictions);
                const mae = this.calculateMAE(Y, predictions);
                const acc = this.calculateAccuracy(Y, predictions);
                const f1 = this.calculateF1Score(Y, predictions);
                const ce = this.calculateCrossEntropy(Y, predictions);
                const r2 = this.calculateR2Score(Y, predictions);
                const results = {};
                let allPassed = true;
                if (this.metrics.rmse !== undefined) {
                    results.rmse = rmse;
                    if (rmse > this.metrics.rmse)
                        allPassed = false;
                }
                if (this.metrics.mae !== undefined) {
                    results.mae = mae;
                    if (mae > this.metrics.mae)
                        allPassed = false;
                }
                if (this.metrics.accuracy !== undefined) {
                    results.accuracy = acc;
                    if (acc < this.metrics.accuracy)
                        allPassed = false;
                }
                if (this.metrics.f1 !== undefined) {
                    results.f1 = f1;
                    if (f1 < this.metrics.f1)
                        allPassed = false;
                }
                if (this.metrics.crossEntropy !== undefined) {
                    results.crossEntropy = ce;
                    if (ce > this.metrics.crossEntropy)
                        allPassed = false;
                }
                if (this.metrics.r2 !== undefined) {
                    results.r2 = r2;
                    if (r2 < this.metrics.r2)
                        allPassed = false;
                }
                if (this.verbose) {
                    this.logMetrics(results);
                }
                if (allPassed) {
                    this.savedModelJSON = JSON.stringify(this.model);
                    if (this.verbose)
                        console.log("✅ Model passed thresholds and was saved to JSON.");
                    if (this.config.exportFileName) {
                        this.saveModelAsJSONFile(this.config.exportFileName);
                    }
                }
                else {
                    if (this.verbose)
                        console.log("❌ Model not saved: One or more thresholds not met.");
                }
            }
            else {
                this.savedModelJSON = JSON.stringify(this.model);
                if (this.verbose)
                    console.log("✅ Model trained with no metrics—saved by default.");
                if (this.config.exportFileName) {
                    this.saveModelAsJSONFile(this.config.exportFileName);
                }
            }
        }
        logMetrics(results) {
            var _a, _b, _c, _d, _e, _f;
            const logLines = [`📋 ${this.modelName} — Metrics Summary:`];
            const push = (label, value, threshold, cmp) => {
                if (threshold !== undefined)
                    logLines.push(`  ${label}: ${value.toFixed(4)} (threshold: ${cmp} ${threshold})`);
            };
            push('RMSE', results.rmse, (_a = this.metrics) === null || _a === void 0 ? void 0 : _a.rmse, '<=');
            push('MAE', results.mae, (_b = this.metrics) === null || _b === void 0 ? void 0 : _b.mae, '<=');
            push('Accuracy', results.accuracy, (_c = this.metrics) === null || _c === void 0 ? void 0 : _c.accuracy, '>=');
            push('F1 Score', results.f1, (_d = this.metrics) === null || _d === void 0 ? void 0 : _d.f1, '>=');
            push('Cross-Entropy', results.crossEntropy, (_e = this.metrics) === null || _e === void 0 ? void 0 : _e.crossEntropy, '<=');
            push('R² Score', results.r2, (_f = this.metrics) === null || _f === void 0 ? void 0 : _f.r2, '>=');
            if (this.verbose)
                console.log('\n' + logLines.join('\n'));
            if (this.logToFile) {
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                const logFile = this.config.logFileName || `${this.modelName.toLowerCase().replace(/\s+/g, '_')}_metrics_${timestamp}.txt`;
                const blob = new Blob([logLines.join('\n')], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = logFile;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
        }
        saveModelAsJSONFile(filename) {
            if (!this.savedModelJSON) {
                if (this.verbose)
                    console.warn("No model saved — did not meet metric thresholds.");
                return;
            }
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const fallback = `${this.modelName.toLowerCase().replace(/\s+/g, '_')}_${timestamp}.json`;
            const finalName = filename || this.config.exportFileName || fallback;
            const blob = new Blob([this.savedModelJSON], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = finalName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            if (this.verbose)
                console.log(`📦 Model exported as ${finalName}`);
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
        calculateRMSE(Y, P) {
            const N = Y.length;
            let sum = 0;
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < Y[0].length; j++) {
                    const diff = Y[i][j] - P[i][j];
                    sum += diff * diff;
                }
            }
            return Math.sqrt(sum / (N * Y[0].length));
        }
        calculateMAE(Y, P) {
            const N = Y.length;
            let sum = 0;
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < Y[0].length; j++) {
                    sum += Math.abs(Y[i][j] - P[i][j]);
                }
            }
            return sum / (N * Y[0].length);
        }
        calculateAccuracy(Y, P) {
            let correct = 0;
            for (let i = 0; i < Y.length; i++) {
                const yMax = Y[i].indexOf(Math.max(...Y[i]));
                const pMax = P[i].indexOf(Math.max(...P[i]));
                if (yMax === pMax)
                    correct++;
            }
            return correct / Y.length;
        }
        calculateF1Score(Y, P) {
            let tp = 0, fp = 0, fn = 0;
            for (let i = 0; i < Y.length; i++) {
                const yIdx = Y[i].indexOf(1);
                const pIdx = P[i].indexOf(Math.max(...P[i]));
                if (yIdx === pIdx)
                    tp++;
                else {
                    fp++;
                    fn++;
                }
            }
            const precision = tp / (tp + fp || 1);
            const recall = tp / (tp + fn || 1);
            return 2 * (precision * recall) / (precision + recall || 1);
        }
        calculateCrossEntropy(Y, P) {
            let loss = 0;
            for (let i = 0; i < Y.length; i++) {
                for (let j = 0; j < Y[0].length; j++) {
                    const pred = Math.min(Math.max(P[i][j], 1e-15), 1 - 1e-15);
                    loss += -Y[i][j] * Math.log(pred);
                }
            }
            return loss / Y.length;
        }
        calculateR2Score(Y, P) {
            const Y_mean = Y[0].map((_, j) => Y.reduce((sum, y) => sum + y[j], 0) / Y.length);
            let ssRes = 0, ssTot = 0;
            for (let i = 0; i < Y.length; i++) {
                for (let j = 0; j < Y[0].length; j++) {
                    ssRes += Math.pow(Y[i][j] - P[i][j], 2);
                    ssTot += Math.pow(Y[i][j] - Y_mean[j], 2);
                }
            }
            return 1 - ssRes / ssTot;
        }
        computeHiddenLayer(X) {
            if (!this.model)
                throw new Error("Model not trained.");
            const WX = Matrix.multiply(X, Matrix.transpose(this.model.W));
            const WXb = WX.map(row => row.map((val, j) => val + this.model.b[j][0]));
            const activationFn = Activations.get(this.activation);
            return WXb.map(row => row.map(activationFn));
        }
        getEmbedding(X) {
            return this.computeHiddenLayer(X);
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
        tokenizerDelimiter: /[\s,.;!?()\[\]{}"']+/,
        log: {}
    };

    // ✅ AutoComplete.ts patched to support (input, label) training and evaluation
    class AutoComplete {
        constructor(pairs, options) {
            var _a;
            this.trainPairs = pairs;
            this.activation = (_a = options.activation) !== null && _a !== void 0 ? _a : 'relu';
            const categories = Array.from(new Set(pairs.map(p => p.label)));
            this.elm = new ELM(Object.assign(Object.assign({}, EnglishTokenPreset), { categories, activation: this.activation, metrics: options.metrics, log: {
                    modelName: "AutoComplete",
                    verbose: options.verbose
                }, exportFileName: options.exportFileName }));
            bindAutocompleteUI({
                model: this.elm,
                inputElement: options.inputElement,
                outputElement: options.outputElement,
                topK: options.topK
            });
        }
        train() {
            const X = [];
            const Y = [];
            for (const { input, label } of this.trainPairs) {
                const vec = this.elm.encoder.normalize(this.elm.encoder.encode(input));
                const labelIndex = this.elm.categories.indexOf(label);
                if (labelIndex === -1)
                    continue;
                X.push(vec);
                Y.push(this.elm.oneHot(this.elm.categories.length, labelIndex));
            }
            this.elm.trainFromData(X, Y);
        }
        predict(input, topN = 1) {
            return this.elm.predict(input, topN).map(p => ({
                completion: p.label,
                prob: p.prob
            }));
        }
        getModel() {
            return this.elm;
        }
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
        top1Accuracy(pairs) {
            var _a;
            let correct = 0;
            for (const { input, label } of pairs) {
                const [pred] = this.predict(input, 1);
                if (((_a = pred === null || pred === void 0 ? void 0 : pred.completion) === null || _a === void 0 ? void 0 : _a.toLowerCase().trim()) === label.toLowerCase().trim()) {
                    correct++;
                }
            }
            return correct / pairs.length;
        }
        crossEntropy(pairs) {
            var _a;
            let totalLoss = 0;
            for (const { input, label } of pairs) {
                const preds = this.predict(input, 5);
                const match = preds.find(p => p.completion.toLowerCase().trim() === label.toLowerCase().trim());
                const prob = (_a = match === null || match === void 0 ? void 0 : match.prob) !== null && _a !== void 0 ? _a : 1e-6;
                totalLoss += -Math.log(prob); // ⬅ switched from log2 to natural log
            }
            return totalLoss / pairs.length;
        }
        internalCrossEntropy(verbose = false) {
            const { model, encoder, categories } = this.elm;
            if (!model) {
                if (verbose)
                    console.warn("⚠️ Cannot compute internal cross-entropy: model not trained.");
                return Infinity;
            }
            const X = [];
            const Y = [];
            for (const { input, label } of this.trainPairs) {
                const vec = encoder.normalize(encoder.encode(input));
                const labelIdx = categories.indexOf(label);
                if (labelIdx === -1)
                    continue;
                X.push(vec);
                Y.push(this.elm.oneHot(categories.length, labelIdx));
            }
            const { W, b, beta } = model;
            const tempH = Matrix.multiply(X, Matrix.transpose(W));
            const activationFn = Activations.get(this.activation);
            const H = Activations.apply(tempH.map(row => row.map((val, j) => val + b[j][0])), activationFn);
            const preds = Matrix.multiply(H, beta);
            const ce = this.elm.calculateCrossEntropy(Y, preds);
            if (verbose) {
                console.log(`📏 Internal Cross-Entropy (full model eval): ${ce.toFixed(4)}`);
            }
            return ce;
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
            this.config = Object.assign(Object.assign({}, config), { categories: [], useTokenizer: true, log: {
                    modelName: "EncoderELM",
                    verbose: config.log.verbose
                } });
            this.elm = new ELM(this.config);
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
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
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    // intentClassifier.ts - ELM-based intent classification engine
    class IntentClassifier {
        constructor(config) {
            this.config = Object.assign(Object.assign({}, config), { log: {
                    modelName: "IntentClassifier",
                    verbose: config.log.verbose
                } });
            this.model = new ELM(config);
            if (config.metrics)
                this.model.metrics = config.metrics;
            if (config.exportFileName)
                this.model.config.exportFileName = config.exportFileName;
        }
        train(textLabelPairs, augmentationOptions) {
            const labelSet = Array.from(new Set(textLabelPairs.map(p => p.label)));
            this.model.setCategories(labelSet);
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
        loadModelFromJSON(json) {
            this.model.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.model.saveModelAsJSONFile(filename);
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
            this.config = Object.assign(Object.assign({}, config), { log: {
                    modelName: "IntentClassifier",
                    verbose: config.log.verbose
                } });
            this.elm = new ELM(config);
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
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
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
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
            this.config = Object.assign(Object.assign({}, config), { categories: [], useTokenizer: false, log: {
                    modelName: "FeatureCombinerELM",
                    verbose: config.log.verbose
                } });
            this.elm = new ELM(this.config);
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
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
            if (!this.config.hiddenUnits || !this.config.activation) {
                throw new Error("FeatureCombinerELM: config.hiddenUnits or activation is undefined.");
            }
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
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    /**
     * VotingClassifierELM takes predictions from multiple ELMs
     * and learns to choose the most accurate final label.
     * It can optionally incorporate confidence scores and calibrate model weights.
     */
    class VotingClassifierELM {
        constructor(config) {
            this.categories = config.categories || ['English', 'French', 'Spanish'];
            this.modelWeights = [];
            this.elm = new ELM(Object.assign(Object.assign({}, config), { useTokenizer: false, categories: this.categories, log: {
                    modelName: "IntentClassifier",
                    verbose: config.log.verbose
                } }));
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
        }
        setModelWeights(weights) {
            this.modelWeights = weights;
        }
        calibrateWeights(predictionLists, trueLabels) {
            const numModels = predictionLists.length;
            const numExamples = trueLabels.length;
            const accuracies = new Array(numModels).fill(0);
            for (let m = 0; m < numModels; m++) {
                let correct = 0;
                for (let i = 0; i < numExamples; i++) {
                    if (predictionLists[m][i] === trueLabels[i]) {
                        correct++;
                    }
                }
                accuracies[m] = correct / numExamples;
            }
            const total = accuracies.reduce((sum, acc) => sum + acc, 0) || 1;
            this.modelWeights = accuracies.map(a => a / total);
            console.log('🔧 Calibrated model weights based on accuracy:', this.modelWeights);
        }
        train(predictionLists, confidenceLists, trueLabels) {
            if (!Array.isArray(predictionLists) || predictionLists.length === 0 || !trueLabels) {
                throw new Error('Invalid inputs to VotingClassifierELM.train');
            }
            const numModels = predictionLists.length;
            const numExamples = predictionLists[0].length;
            for (let list of predictionLists) {
                if (list.length !== numExamples) {
                    throw new Error('Inconsistent prediction lengths across models');
                }
            }
            if (confidenceLists) {
                if (confidenceLists.length !== numModels) {
                    throw new Error('Confidence list count must match number of models');
                }
                for (let list of confidenceLists) {
                    if (list.length !== numExamples) {
                        throw new Error('Inconsistent confidence lengths across models');
                    }
                }
            }
            if (!this.modelWeights || this.modelWeights.length !== numModels) {
                this.calibrateWeights(predictionLists, trueLabels);
            }
            const inputs = [];
            for (let i = 0; i < numExamples; i++) {
                let inputRow = [];
                for (let m = 0; m < numModels; m++) {
                    const label = predictionLists[m][i];
                    if (typeof label === 'undefined') {
                        console.error(`Undefined label from model ${m} at index ${i}`);
                        throw new Error(`Invalid label in predictionLists[${m}][${i}]`);
                    }
                    const weight = this.modelWeights[m];
                    inputRow = inputRow.concat(this.oneHot(label).map(x => x * weight));
                    if (confidenceLists) {
                        const conf = confidenceLists[m][i];
                        const normalizedConf = Math.min(1, Math.max(0, conf));
                        inputRow.push(normalizedConf * weight);
                    }
                }
                inputs.push(inputRow);
            }
            const examples = inputs.map((input, i) => ({ input, label: trueLabels[i] }));
            console.log(`📊 VotingClassifierELM training on ${examples.length} examples with ${numModels} models.`);
            this.elm.train(examples);
        }
        predict(labels, confidences) {
            if (!Array.isArray(labels) || labels.length === 0) {
                throw new Error('No labels provided to VotingClassifierELM.predict');
            }
            let input = [];
            for (let i = 0; i < labels.length; i++) {
                const weight = this.modelWeights[i] || 1;
                input = input.concat(this.oneHot(labels[i]).map(x => x * weight));
                if (confidences && typeof confidences[i] === 'number') {
                    const norm = Math.min(1, Math.max(0, confidences[i]));
                    input.push(norm * weight);
                }
            }
            return this.elm.predict(JSON.stringify(input), 1);
        }
        oneHot(label) {
            const index = this.categories.indexOf(label);
            if (index === -1) {
                console.warn(`Unknown label in oneHot: ${label}`);
                return new Array(this.categories.length).fill(0);
            }
            return this.categories.map((_, i) => (i === index ? 1 : 0));
        }
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    /**
     * ConfidenceClassifierELM is a lightweight ELM wrapper
     * designed to classify whether an input prediction is likely to be high or low confidence.
     * It uses the same input format as FeatureCombinerELM (vector + meta).
     */
    class ConfidenceClassifierELM {
        constructor(config) {
            this.config = config;
            this.elm = new ELM(Object.assign(Object.assign({}, config), { categories: ['low', 'high'], useTokenizer: false, log: {
                    modelName: "ConfidenceClassifierELM",
                    verbose: config.log.verbose
                } }));
            // Forward optional ELM config extensions
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
        }
        train(vectors, metas, labels) {
            vectors.map((vec, i) => FeatureCombinerELM.combineFeatures(vec, metas[i]));
            const examples = vectors.map((vec, i) => ({
                input: FeatureCombinerELM.combineFeatures(vec, metas[i]),
                label: labels[i]
            }));
            this.elm.train(examples);
        }
        predict(vec, meta) {
            const input = FeatureCombinerELM.combineFeatures(vec, meta);
            const inputStr = JSON.stringify(input);
            return this.elm.predict(inputStr, 1);
        }
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    class RefinerELM {
        constructor(config) {
            this.config = Object.assign(Object.assign({}, config), { useTokenizer: false, categories: [], log: {
                    modelName: "IntentClassifier",
                    verbose: config.log.verbose
                } });
            this.elm = new ELM(this.config);
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
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
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    class CharacterLangEncoderELM {
        constructor(config) {
            if (!config.hiddenUnits || !config.activation) {
                throw new Error("CharacterLangEncoderELM requires defined hiddenUnits and activation");
            }
            this.config = Object.assign(Object.assign({}, config), { log: {
                    modelName: "CharacterLangEncoderELM",
                    verbose: config.log.verbose
                }, useTokenizer: true });
            this.elm = new ELM(this.config);
            // Forward ELM-specific options
            if (config.metrics)
                this.elm.metrics = config.metrics;
            if (config.exportFileName)
                this.elm.config.exportFileName = config.exportFileName;
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
        loadModelFromJSON(json) {
            this.elm.loadModelFromJSON(json);
        }
        saveModelAsJSONFile(filename) {
            this.elm.saveModelAsJSONFile(filename);
        }
    }

    class ModulePool {
        constructor() {
            this.modules = [];
        }
        addModule(mod) {
            this.modules.push(mod);
        }
        removeModule(id) {
            this.modules = this.modules.filter((m) => m.id !== id);
        }
        getModuleById(id) {
            return this.modules.find((m) => m.id === id);
        }
        listModules() {
            return this.modules;
        }
        evolveModules() {
            const clones = [];
            const survivors = [];
            for (const mod of this.modules) {
                const score = this.computeScore(mod.metrics);
                if (score < 0.3) {
                    console.log(`❌ Removing ${mod.id} (score ${score.toFixed(4)})`);
                    continue;
                }
                if (score > 0.6) {
                    const newId = `${mod.id}-clone-${Date.now()}`;
                    console.log(`✨ Cloning ${mod.id} -> ${newId}`);
                    const cloned = Object.assign(Object.assign({}, mod), { id: newId });
                    // Simple mutation
                    if (cloned.elm && cloned.elm.config && typeof cloned.elm.config.dropout === "number") {
                        cloned.elm.config.dropout = Math.max(0, Math.min(0.2, cloned.elm.config.dropout + (Math.random() * 0.02 - 0.01)));
                    }
                    clones.push(cloned);
                }
                survivors.push(mod);
            }
            this.modules = [...survivors, ...clones];
        }
        computeScore(metrics) {
            return (metrics.recall1 * 0.5 +
                metrics.recall5 * 0.3 +
                metrics.mrr * 0.2 -
                metrics.avgLatency * 0.01);
        }
        /**
         * Each module emits a signal to the bus.
         */
        broadcastAllSignals(bus, vectorizer) {
            for (const mod of this.modules) {
                const text = `Signal from ${mod.id}`;
                const vector = vectorizer.vectorize(text);
                bus.broadcast({
                    id: `${mod.id}-${Date.now()}`,
                    sourceModuleId: mod.id,
                    vector,
                    timestamp: Date.now(),
                    metadata: {
                        role: mod.role
                    }
                });
            }
        }
        /**
         * Each module consumes signals from others.
         */
        consumeAllSignals(bus) {
            for (const mod of this.modules) {
                const signals = bus.getSignalsFromOthers(mod.id);
                // You can define how to consume them.
                // For example, count how many signals were observed:
                if (signals.length > 0) {
                    console.log(`🔊 ${mod.id} observed ${signals.length} signals from peers.`);
                }
            }
        }
    }

    // src/core/SignalBus.ts
    class SignalBus {
        constructor(maxHistory = 500) {
            this.signals = [];
            this.maxHistory = maxHistory;
        }
        // Broadcast a new signal
        broadcast(signal) {
            this.signals.push(signal);
            if (this.signals.length > this.maxHistory) {
                this.signals.shift(); // remove oldest
            }
        }
        // Retrieve all signals
        getSignals() {
            return this.signals.slice();
        }
        // Retrieve signals from other modules only
        getSignalsFromOthers(moduleId) {
            return this.signals.filter(s => s.sourceModuleId !== moduleId);
        }
        // Clear all signals (optional)
        clear() {
            this.signals = [];
        }
    }

    // MiniTransformer.ts
    class Vocab {
        constructor(tokens) {
            this.tokens = tokens;
            this.tokenToIdx = {};
            this.idxToToken = {};
            tokens.forEach((t, i) => {
                this.tokenToIdx[t] = i;
                this.idxToToken[i] = t;
            });
        }
        encode(text) {
            return text
                .split("")
                .map(c => this.tokenToIdx[c])
                .filter((idx) => idx !== undefined);
        }
    }
    class MiniTransformer {
        constructor(vocab) {
            this.vocab = vocab;
            this.embedDim = 32;
            this.seqLen = 16;
            this.embedding = this.randomMatrix(vocab.tokens.length, this.embedDim);
            this.posEnc = this.positionalEncoding(this.seqLen, this.embedDim);
        }
        randomMatrix(rows, cols) {
            return Array.from({ length: rows }, () => Array.from({ length: cols }, () => Math.random() * 2 - 1));
        }
        positionalEncoding(seqLen, embedDim) {
            const encoding = [];
            for (let pos = 0; pos < seqLen; pos++) {
                const row = [];
                for (let i = 0; i < embedDim; i++) {
                    const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / embedDim);
                    row.push(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
                }
                encoding.push(row);
            }
            return encoding;
        }
        encode(inputIndices) {
            if (!inputIndices || inputIndices.length === 0) {
                // Fallback: zero vector
                return Array(this.embedDim).fill(0);
            }
            // For each token index, get embedding + position encoding
            const x = inputIndices.map((idx, i) => {
                if (idx === undefined) {
                    // Unknown token fallback embedding
                    return Array.from({ length: this.embedDim }, () => 0);
                }
                return this.embedding[idx].map((v, j) => v + this.posEnc[i % this.seqLen][j]);
            });
            // Aggregate into a single embedding (mean)
            return this.meanVecs(x);
        }
        meanVecs(vecs) {
            const out = Array(vecs[0].length).fill(0);
            for (const vec of vecs) {
                for (let i = 0; i < vec.length; i++) {
                    out[i] += vec[i];
                }
            }
            return out.map(v => v / vecs.length);
        }
    }

    class TFIDF {
        constructor(corpusDocs, options = {}) {
            this.termFrequency = {};
            this.inverseDocFreq = {};
            this.wordsInDoc = [];
            this.processedWords = [];
            this.scores = {};
            this.corpus = "";
            this.options = options;
            this.corpus = corpusDocs.join(" ");
            const wordsFinal = [];
            const re = /[^a-zA-Z0-9]+/g;
            corpusDocs.forEach(doc => {
                const tokens = doc.split(/\s+/);
                tokens.forEach(word => {
                    const cleaned = word.replace(re, " ");
                    wordsFinal.push(...cleaned.split(/\s+/).filter(Boolean));
                });
            });
            this.wordsInDoc = wordsFinal;
            this.processedWords = TFIDF.processWords(wordsFinal, options);
            // Compute term frequency
            this.processedWords.forEach(token => {
                this.termFrequency[token] = (this.termFrequency[token] || 0) + 1;
            });
            // Compute inverse document frequency
            for (const term in this.termFrequency) {
                const count = TFIDF.countDocsContainingTerm(corpusDocs, term);
                this.inverseDocFreq[term] = Math.log(corpusDocs.length / (1 + count));
            }
        }
        static countDocsContainingTerm(corpusDocs, term) {
            return corpusDocs.reduce((acc, doc) => (doc.includes(term) ? acc + 1 : acc), 0);
        }
        static processWords(words, options = {}) {
            const filtered = TFIDF.removeStopWordsAndStem(words, options).map(w => TFIDF.lemmatize(w, options));
            const bigrams = TFIDF.generateNGrams(filtered, 2);
            const trigrams = TFIDF.generateNGrams(filtered, 3);
            return [...filtered, ...bigrams, ...trigrams];
        }
        static removeStopWordsAndStem(words, options = {}) {
            var _a;
            const defaultStopWords = new Set([
                "a", "and", "the", "is", "to", "of", "in", "it", "that", "you",
                "this", "for", "on", "are", "with", "as", "be", "by", "at", "from",
                "or", "an", "but", "not", "we"
            ]);
            const stopWords = (_a = options.stopWords) !== null && _a !== void 0 ? _a : defaultStopWords;
            return words.filter(w => !stopWords.has(w)).map(w => TFIDF.advancedStem(w));
        }
        static advancedStem(word) {
            const suffixes = ["es", "ed", "ing", "s", "ly", "ment", "ness", "ity", "ism", "er"];
            for (const suffix of suffixes) {
                if (word.endsWith(suffix)) {
                    if (suffix === "es" && word.length > 2 && word[word.length - 3] === "i") {
                        return word.slice(0, -2);
                    }
                    return word.slice(0, -suffix.length);
                }
            }
            return word;
        }
        static lemmatize(word, options = {}) {
            if (options.lemmatizationRules && options.lemmatizationRules[word]) {
                return options.lemmatizationRules[word];
            }
            if (word.endsWith("ing"))
                return word.slice(0, -3);
            if (word.endsWith("ed"))
                return word.slice(0, -2);
            return word;
        }
        static generateNGrams(tokens, n) {
            if (tokens.length < n)
                return [];
            const ngrams = [];
            for (let i = 0; i <= tokens.length - n; i++) {
                ngrams.push(tokens.slice(i, i + n).join(" "));
            }
            return ngrams;
        }
        calculateScores() {
            const totalWords = this.processedWords.length;
            const scores = {};
            this.processedWords.forEach(token => {
                const tf = this.termFrequency[token] || 0;
                scores[token] = (tf / totalWords) * (this.inverseDocFreq[token] || 0);
            });
            this.scores = scores;
            return scores;
        }
        extractKeywords(topN) {
            const entries = Object.entries(this.scores).sort((a, b) => b[1] - a[1]);
            return Object.fromEntries(entries.slice(0, topN));
        }
        processedWordsIndex(word) {
            return this.processedWords.indexOf(word);
        }
    }
    class TFIDFVectorizer {
        constructor(docs, options = {}) {
            var _a;
            this.docTexts = docs;
            this.tfidf = new TFIDF(docs, options);
            // Collect all unique terms
            const termFreq = {};
            docs.forEach(doc => {
                const tokens = doc.split(/\s+/);
                const cleaned = tokens.map(t => t.replace(/[^a-zA-Z0-9]+/g, ""));
                const processed = TFIDF.processWords(cleaned, options);
                processed.forEach(t => {
                    termFreq[t] = (termFreq[t] || 0) + 1;
                });
            });
            const maxVocab = (_a = options.maxVocabSize) !== null && _a !== void 0 ? _a : 2000;
            const sortedTerms = Object.entries(termFreq)
                .sort((a, b) => b[1] - a[1])
                .slice(0, maxVocab)
                .map(([term]) => term);
            this.vocabulary = sortedTerms;
            console.log(`✅ TFIDFVectorizer vocabulary capped at: ${this.vocabulary.length} terms.`);
        }
        vectorize(doc) {
            const tokens = doc.split(/\s+/);
            const cleaned = tokens.map(t => t.replace(/[^a-zA-Z0-9]+/g, ""));
            const processed = TFIDF.processWords(cleaned, this.tfidf.options);
            const termFreq = {};
            processed.forEach(token => {
                termFreq[token] = (termFreq[token] || 0) + 1;
            });
            const totalTerms = processed.length;
            return this.vocabulary.map(term => {
                const tf = totalTerms > 0 ? (termFreq[term] || 0) / totalTerms : 0;
                const idf = this.tfidf.inverseDocFreq[term] || 0;
                return tf * idf;
            });
        }
        vectorizeAll() {
            return this.docTexts.map(doc => this.vectorize(doc));
        }
        static l2normalize(vec) {
            const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
            return norm === 0 ? vec : vec.map(x => x / norm);
        }
    }

    exports.Activations = Activations;
    exports.Augment = Augment;
    exports.AutoComplete = AutoComplete;
    exports.CharacterLangEncoderELM = CharacterLangEncoderELM;
    exports.ConfidenceClassifierELM = ConfidenceClassifierELM;
    exports.ELM = ELM;
    exports.EncoderELM = EncoderELM;
    exports.FeatureCombinerELM = FeatureCombinerELM;
    exports.IO = IO;
    exports.IntentClassifier = IntentClassifier;
    exports.LanguageClassifier = LanguageClassifier;
    exports.MiniTransformer = MiniTransformer;
    exports.ModulePool = ModulePool;
    exports.RefinerELM = RefinerELM;
    exports.SignalBus = SignalBus;
    exports.TFIDF = TFIDF;
    exports.TFIDFVectorizer = TFIDFVectorizer;
    exports.TextEncoder = TextEncoder;
    exports.Tokenizer = Tokenizer;
    exports.UniversalEncoder = UniversalEncoder;
    exports.Vocab = Vocab;
    exports.VotingClassifierELM = VotingClassifierELM;
    exports.bindAutocompleteUI = bindAutocompleteUI;
    exports.defaultConfig = defaultConfig;

}));
//# sourceMappingURL=astermind.umd.js.map
