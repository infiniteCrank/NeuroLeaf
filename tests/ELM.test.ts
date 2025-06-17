import { describe, it, expect, beforeEach } from 'vitest';
import { ELM } from '../src/core/ELM';
import { ELMConfig } from '../src/core/ELMConfig';

describe('ELM', () => {
    let model: ELM;

    const genres = ['rock', 'pop', 'jazz'];

    const config: ELMConfig = {
        categories: genres,
        hiddenUnits: 10,
        maxLen: 10,
        activation: 'relu',
    };

    beforeEach(() => {
        model = new ELM(config);
    });

    it('should instantiate with correct categories', () => {
        expect(model.categories).toEqual(genres);
        expect(model.hiddenUnits).toBe(10);
        expect(model.activation).toBe('relu');
    });

    it('throws when predict is called before train()', () => {
        expect(() => model.predict('rock')).toThrowError(/not trained/i);
    });

    it('can train and predict top categories', () => {
        model.train(); // without augmentation

        const predictions = model.predict('roc', 2);
        expect(predictions.length).toBeLessThanOrEqual(2);

        predictions.forEach(pred => {
            expect(pred.label).toBeTypeOf('string');
            expect(pred.prob).toBeGreaterThan(0);
        });

        const probSum = predictions.reduce((acc, p) => acc + p.prob, 0);
        expect(probSum).toBeLessThanOrEqual(1.0); // topK may truncate total
    });

    it('setCategories updates the category list and retrains', () => {
        const newLabels = ['metal', 'funk'];
        model.setCategories(newLabels);
        model.train();

        const prediction = model.predict('funk')[0];
        expect(newLabels.includes(prediction.label)).toBe(true);
    });

    it('respects activation function (leakyReLU)', () => {
        const leakyConfig = { ...config, activation: 'leakyrelu' };
        const leakyModel = new ELM(leakyConfig);
        leakyModel.train();
        const result = leakyModel.predict('rock');
        expect(result.length).toBeGreaterThan(0);
    });
});
