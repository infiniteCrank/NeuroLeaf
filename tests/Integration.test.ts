// Integration.test.ts - Integration tests for core components

import { describe, it, expect } from 'vitest';
import { LanguageClassifier } from '../src/tasks/LanguageClassifier';
import { ELMConfig } from '../src/core/ELMConfig';

const config: ELMConfig = {
    categories: ['english', 'spanish'],
    hiddenUnits: 10,
    maxLen: 10,
    activation: 'sigmoid',
    charSet: 'abcdefghijklmnopqrstuvwxyz ',
};

describe('Integration: LanguageClassifier pipeline', () => {
    const classifier = new LanguageClassifier(config);

    it('trains and predicts from JSON data', () => {
        const trainingDataJSON = JSON.stringify([
            { text: 'hello', label: 'english' },
            { text: 'hi there', label: 'english' },
            { text: 'hola', label: 'spanish' },
            { text: 'buenos dias', label: 'spanish' },
        ]);

        const data = classifier.loadTrainingData(trainingDataJSON, 'json');
        classifier.train(data);

        const prediction = classifier.predict('hola');
        expect(prediction.length).toBeGreaterThan(0);
        expect(prediction[0].label).toBe('spanish');
        expect(prediction[0].prob).toBeGreaterThan(0.5);
    });

    it('trains and predicts from TSV data', () => {
        const trainingTSV = `text\tlabel\nhello\tenglish\nhola\tspanish`;
        const data = classifier.loadTrainingData(trainingTSV, 'tsv');
        classifier.train(data);
        const prediction = classifier.predict('hello');
        expect(prediction[0].label).toBe('english');
    });

    it('predicts with topK and returns probabilities', () => {
        const result = classifier.predict('buenos dias', 2);
        expect(result.length).toBeLessThanOrEqual(2);
        expect(result[0]).toHaveProperty('label');
        expect(result[0]).toHaveProperty('prob');
    });
});
