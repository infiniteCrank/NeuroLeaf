export class Matrix {
    private data: number[][];
    rows: number;
    cols: number;

    constructor(rows: number, cols: number, fill = 0) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => fill)
        );
    }

    static from2D(values: number[][]): Matrix {
        const m = new Matrix(values.length, values[0].length);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.set(i, j, values[i][j]);
            }
        }
        return m;
    }

    toArray(): number[][] {
        return this.data.map(row => [...row]);
    }

    get(i: number, j: number): number {
        return this.data[i][j];
    }

    set(i: number, j: number, value: number): void {
        this.data[i][j] = value;
    }

    map(fn: (x: number) => number): Matrix {
        return Matrix.from2D(this.data.map(row => row.map(fn)));
    }

    add(other: Matrix): Matrix {
        this.checkShapeMatch(other);
        return Matrix.from2D(
            this.data.map((row, i) => row.map((val, j) => val + other.get(i, j)))
        );
    }

    subtract(other: Matrix): Matrix {
        this.checkShapeMatch(other);
        return Matrix.from2D(
            this.data.map((row, i) => row.map((val, j) => val - other.get(i, j)))
        );
    }

    elementwiseMultiply(other: Matrix): Matrix {
        this.checkShapeMatch(other);
        return Matrix.from2D(
            this.data.map((row, i) => row.map((val, j) => val * other.get(i, j)))
        );
    }

    multiply(other: Matrix): Matrix {
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

    dot(other: Matrix): Matrix {
        return this.multiply(other);
    }

    transpose(): Matrix {
        const result = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.set(j, i, this.get(i, j));
            }
        }
        return result;
    }

    inverse(): Matrix {
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
                if (swapRow === -1) throw new Error("Matrix is singular.");
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

    static identity(n: number): Matrix {
        const m = new Matrix(n, n);
        for (let i = 0; i < n; i++) {
            m.set(i, i, 1);
        }
        return m;
    }

    static random(rows: number, cols: number, min = -1, max = 1): Matrix {
        const m = new Matrix(rows, cols);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                m.set(i, j, Math.random() * (max - min) + min);
            }
        }
        return m;
    }

    private checkShapeMatch(other: Matrix) {
        if (this.rows !== other.rows || this.cols !== other.cols) {
            throw new Error("Matrix shape mismatch.");
        }
    }

    reshape(newRows: number, newCols: number): Matrix {
        if (newRows * newCols !== this.rows * this.cols) {
            throw new Error("New shape must match total number of elements.");
        }
        const flat = this.data.flat();
        const reshaped: number[][] = [];
        for (let i = 0; i < newRows; i++) {
            reshaped.push(flat.slice(i * newCols, (i + 1) * newCols));
        }
        return Matrix.from2D(reshaped);
    }

    sum(axis?: 0 | 1): number[] | number {
        if (axis === 0) {
            // column-wise sum
            return Array.from({ length: this.cols }, (_, j) =>
                this.data.reduce((sum, row) => sum + row[j], 0)
            );
        } else if (axis === 1) {
            // row-wise sum
            return this.data.map(row => row.reduce((a, b) => a + b, 0));
        } else {
            // full sum
            return this.data.flat().reduce((a, b) => a + b, 0);
        }
    }

    mean(axis?: 0 | 1): number[] | number {
        const sumResult = this.sum(axis);

        if (axis === 0 && Array.isArray(sumResult)) {
            return sumResult.map(sum => sum / this.rows);
        } else if (axis === 1 && Array.isArray(sumResult)) {
            return sumResult.map(sum => sum / this.cols);
        } else if (typeof sumResult === 'number') {
            return sumResult / (this.rows * this.cols);
        } else {
            throw new Error("Unexpected shape in mean()");
        }
    }

    flatten(): number[] {
        return this.data.flat();
    }

    argmax(): { row: number; col: number; value: number } {
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

    softmax(): Matrix {
        const flat = this.flatten();
        const max = Math.max(...flat); // for numerical stability
        const exps = flat.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        const softmaxValues = exps.map(x => x / sum);

        // Return as 1-row matrix
        return Matrix.from2D([softmaxValues]);
    }

    clip(min: number, max: number): Matrix {
        return this.map(x => Math.max(min, Math.min(max, x)));
    }

    normalize(): Matrix {
        const flat = this.flatten();
        const norm = Math.sqrt(flat.reduce((sum, x) => sum + x * x, 0)) || 1;
        return this.map(x => x / norm);
    }

    standardize(): Matrix {
        const flat = this.flatten();
        const mean = flat.reduce((sum, x) => sum + x, 0) / flat.length;
        const variance = flat.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / flat.length;
        const std = Math.sqrt(variance) || 1;

        return this.map(x => (x - mean) / std);
    }

    minMaxScale(): Matrix {
        const flat = this.flatten();
        const min = Math.min(...flat);
        const max = Math.max(...flat);
        const range = max - min || 1;

        return this.map(x => (x - min) / range);
    }

    rowWiseNormalize(): Matrix {
        const newRows = this.toArray().map(row => {
            const norm = Math.sqrt(row.reduce((sum, x) => sum + x * x, 0)) || 1;
            return row.map(x => x / norm);
        });
        return Matrix.from2D(newRows);
    }

    zscore(axis?: 0 | 1): Matrix {
        const rows = this.rows;
        const cols = this.cols;

        if (axis === 0) {
            const means = this.mean(0) as number[];
            const stds = this.toArray()[0].map((_, j) => {
                const col = this.data.map(row => row[j]);
                const mean = means[j];
                const variance = col.reduce((sum, x) => sum + (x - mean) ** 2, 0) / rows;
                return Math.sqrt(variance) || 1;
            });

            const z = this.toArray().map(row =>
                row.map((val, j) => (val - means[j]) / stds[j])
            );
            return Matrix.from2D(z);
        } else if (axis === 1) {
            const z = this.toArray().map(row => {
                const mean = row.reduce((a, b) => a + b, 0) / cols;
                const std = Math.sqrt(row.reduce((a, b) => a + (b - mean) ** 2, 0) / cols) || 1;
                return row.map(x => (x - mean) / std);
            });
            return Matrix.from2D(z);
        } else {
            return this.standardize();
        }
    }

    describe(axis: 0 | 1 = 0): {
        min: number[];
        max: number[];
        mean: number[];
        std: number[];
        median: number[];
    } {
        const data = this.toArray();
        const dim = axis === 0 ? this.cols : this.rows;
        const out: {
            min: number[];
            max: number[];
            mean: number[];
            std: number[];
            median: number[];
        } = {
            min: [],
            max: [],
            mean: [],
            std: [],
            median: []
        };

        for (let i = 0; i < dim; i++) {
            const values: number[] =
                axis === 0 ? data.map(row => row[i]) : data[i];

            const sorted = [...values].sort((a, b) => a - b);
            const mid = Math.floor(sorted.length / 2);
            const median =
                sorted.length % 2 === 0
                    ? (sorted[mid - 1] + sorted[mid]) / 2
                    : sorted[mid];

            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const std = Math.sqrt(
                values.reduce((sum, x) => sum + (x - mean) ** 2, 0) / values.length
            );

            out.min.push(Math.min(...values));
            out.max.push(Math.max(...values));
            out.mean.push(mean);
            out.std.push(std);
            out.median.push(median);
        }

        // Export to console for dev inspection
        if (typeof window !== 'undefined') {
            console.log('[Matrix.describe]', { axis, ...out });
        }

        return out;
    }

    static inverse(m: Matrix): Matrix {
        return m.inverse();
    }

    static transpose(m: Matrix): Matrix {
        return m.transpose()
    }

    static addRegularization(matrix: number[][], lambda: number): number[][] {
        return matrix.map((row, i) =>
            row.map((val, j) => (i === j ? val + lambda : val))
        );
    }

    static addRegularizationMatrix(m: Matrix, lambda: number): Matrix {
        const updated = m.toArray().map((row, i) =>
            row.map((val, j) => (i === j ? val + lambda : val))
        );
        return Matrix.from2D(updated);
    }

    static mean(m: Matrix, axis?: 0 | 1): number[] | number {
        return m.mean(axis);
    }

    static map(m: Matrix, fn: (x: number) => number): Matrix {
        return m.map(fn);
    }

    static multiply(a: Matrix, b: Matrix): Matrix {
        return a.multiply(b);
    }

}
