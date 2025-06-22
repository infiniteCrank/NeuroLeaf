import { describe, it, expect } from 'vitest';
import { Matrix } from '../src/core/Matrix';

describe('Matrix', () => {
    it('multiply() computes correct matrix product', () => {
        const A = Matrix.from2D([
            [1, 2],
            [3, 4]
        ]);
        const B = Matrix.from2D([
            [5, 6],
            [7, 8]
        ]);
        const result = Matrix.multiply(A, B).toArray();
        expect(result).toEqual([
            [19, 22],
            [43, 50]
        ]);
    });

    it('transpose() returns the transposed matrix', () => {
        const A = Matrix.from2D([
            [1, 2, 3],
            [4, 5, 6]
        ]);
        const result = Matrix.transpose(A).toArray();
        expect(result).toEqual([
            [1, 4],
            [2, 5],
            [3, 6]
        ]);
    });

    it('identity() returns an identity matrix of correct size', () => {
        const I = Matrix.identity(3);
        expect(I).toEqual([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]);
    });

    it('addRegularization() correctly adds lambda to diagonal', () => {
        const A = [
            [1, 2],
            [3, 4]
        ];
        const result = Matrix.addRegularization(A, 0.1);
        expect(result).toEqual([
            [1.1, 2],
            [3, 4.1]
        ]);
    });

    it('inverse() correctly inverts a simple matrix', () => {
        const A = Matrix.from2D([
            [4, 7],
            [2, 6]
        ]);
        const inv = Matrix.inverse(A).toArray();

        const expected = [
            [0.6, -0.7],
            [-0.2, 0.4]
        ];

        for (let i = 0; i < expected.length; i++) {
            for (let j = 0; j < expected[i].length; j++) {
                expect(inv[i][j]).toBeCloseTo(expected[i][j]);
            }
        }
    });

    it('inverse() throws on singular matrix', () => {
        const singular = Matrix.from2D([
            [1, 2],
            [2, 4]
        ]);
        expect(() => singular.inverse()).toThrow(/singular/i);
    });

    it('multiply() handles identity matrix correctly', () => {
        const A = Matrix.from2D([
            [5, 6],
            [7, 8]
        ]);
        const I = Matrix.identity(2);
        expect(Matrix.multiply(A, I).toArray()).toEqual(A.toArray());
        expect(Matrix.multiply(I, A).toArray()).toEqual(A.toArray());
    });

    it('flatten() returns all elements in row-major order', () => {
        const M = Matrix.from2D([[1, 2], [3, 4]]);
        expect(M.flatten()).toEqual([1, 2, 3, 4]);
    });

    it('argmax() returns the correct max element and position', () => {
        const M = Matrix.from2D([[1, 5], [3, 2]]);
        const result = M.argmax();
        expect(result.value).toBe(5);
        expect(result.row).toBe(0);
        expect(result.col).toBe(1);
    });

    it('softmax() returns probabilities that sum to 1', () => {
        const M = Matrix.from2D([[1, 2, 3]]);
        const SM = M.softmax().flatten();
        const sum = SM.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(1, 5);
    });

    it('clip() limits values between min and max', () => {
        const M = Matrix.from2D([[1, 5], [10, -3]]);
        const clipped = M.clip(0, 5).toArray();
        expect(clipped).toEqual([
            [1, 5],
            [5, 0]
        ]);
    });

    it('normalize() returns unit-norm vector', () => {
        const M = Matrix.from2D([[3, 4]]);
        const normed = M.normalize().flatten();
        const magnitude = Math.sqrt(normed.reduce((sum, x) => sum + x * x, 0));
        expect(magnitude).toBeCloseTo(1);
    });

    it('standardize() has mean ~0 and std ~1', () => {
        const M = Matrix.from2D([[1, 2, 3], [4, 5, 6]]);
        const Z = M.standardize().flatten();
        const mean = Z.reduce((a, b) => a + b, 0) / Z.length;
        const variance = Z.reduce((a, b) => a + b ** 2, 0) / Z.length;
        const std = Math.sqrt(variance);
        expect(mean).toBeCloseTo(0, 5);
        expect(std).toBeCloseTo(1, 5);
    });

    it('rowWiseNormalize() returns rows with L2 norm of 1', () => {
        const M = Matrix.from2D([[3, 4], [1, 1]]);
        const R = M.rowWiseNormalize().toArray();
        for (const row of R) {
            const norm = Math.sqrt(row.reduce((sum, x) => sum + x ** 2, 0));
            expect(norm).toBeCloseTo(1);
        }
    });

    it('minMaxScale() scales matrix between 0 and 1', () => {
        const M = Matrix.from2D([[10, 20], [30, 40]]);
        const scaled = M.minMaxScale().flatten();
        const min = Math.min(...scaled);
        const max = Math.max(...scaled);
        expect(min).toBeCloseTo(0);
        expect(max).toBeCloseTo(1);
    });

    it('zscore(axis=0) normalizes each column', () => {
        const M = Matrix.from2D([[1, 2], [3, 4]]);
        const Z = M.zscore(0).toArray();
        expect(Z.length).toBe(2);
        expect(Z[0].length).toBe(2);
    });

    it('zscore(axis=1) normalizes each row', () => {
        const M = Matrix.from2D([[1, 2], [3, 4]]);
        const Z = M.zscore(1).toArray();
        expect(Z.length).toBe(2);
        expect(Z[0].length).toBe(2);
    });

    it('describe(axis=0) returns correct structure', () => {
        const M = Matrix.from2D([[1, 2], [3, 4]]);
        const desc = M.describe(0);
        expect(desc.mean.length).toBe(2);
        expect(desc.min.length).toBe(2);
        expect(desc.max.length).toBe(2);
        expect(desc.median.length).toBe(2);
        expect(desc.std.length).toBe(2);
    });

    it('describe(axis=1) returns correct row stats', () => {
        const M = Matrix.from2D([[1, 2], [3, 4]]);
        const desc = M.describe(1);
        expect(desc.mean.length).toBe(2);
        expect(desc.median.length).toBe(2);
    });

});
