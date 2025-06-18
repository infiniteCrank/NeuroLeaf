// Integration.test.ts - Integration tests for core components

import { describe, it, expect } from 'vitest';
import { LanguageClassifier } from '../src/tasks/LanguageClassifier';
import { ELMConfig } from '../src/core/ELMConfig';
import { AutoComplete } from '../src/tasks/AutoComplete';
import { EnglishCharPreset, RussianTokenPreset, EmojiHybridPreset } from '../src/config/Presets';

const config: ELMConfig = {
    categories: ['english', 'spanish'],
    hiddenUnits: 10,
    maxLen: 10,
    activation: 'sigmoid',
    charSet: 'abcdefghijklmnopqrstuvwxyz ',
};

describe('Integration: LanguageClassifier pipeline', () => {
    it('trains and predicts from JSON data', () => {
        const classifier = new LanguageClassifier(config);
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
        const labels = prediction.map(p => p.label);
        expect(labels).toContain('spanish');
        expect(prediction[0].prob).toBeGreaterThan(0.5);
    });

    it('trains and predicts from TSV data', () => {
        const classifier = new LanguageClassifier(config);
        const trainingTSV = `text\tlabel\nhola\tspanish\nhello\tenglish`;
        const data = classifier.loadTrainingData(trainingTSV, 'tsv');
        classifier.train(data);
        const prediction = classifier.predict('hello');
        const labels = prediction.map(p => p.label);
        expect(labels).toContain('english');
    });

    it('predicts with topK and returns probabilities', () => {
        const classifier = new LanguageClassifier(config);
        const trainingData = [
            { text: 'hello', label: 'english' },
            { text: 'hola', label: 'spanish' }
        ];
        classifier.train(trainingData);
        const result = classifier.predict('buenos dias', 2);
        expect(result.length).toBeLessThanOrEqual(2);
        expect(result[0]).toHaveProperty('label');
        expect(result[0]).toHaveProperty('prob');
    });
});

// AutoComplete

describe('AutoComplete integration', () => {
    it('initializes and binds autocomplete correctly', () => {
        const input = document.createElement('input');
        const output = document.createElement('div');
        const ac = new AutoComplete(['rock', 'pop'], { inputElement: input, outputElement: output });
        input.value = 'roc';
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toContain('rock');
    });
});

// Preset Configuration Tests

describe('ELMConfig Preset Configuration', () => {
    it('has correct defaults for EnglishCharPreset', () => {
        expect(EnglishCharPreset.charSet).toContain('a');
        expect(EnglishCharPreset.useTokenizer).toBe(false);
        expect(EnglishCharPreset.activation).toBeDefined();
    });

    it('has correct configuration for RussianTokenPreset', () => {
        expect(RussianTokenPreset.charSet).toContain('а');
        expect(RussianTokenPreset.useTokenizer).toBe(true);
        expect(RussianTokenPreset.tokenizerDelimiter).toBeInstanceOf(RegExp);
    });

    it('has correct configuration for EmojiHybridPreset', () => {
        expect(EmojiHybridPreset.charSet).toMatch(/\p{Emoji}/u);
        expect(EmojiHybridPreset.useTokenizer).toBe(true);
        expect(EmojiHybridPreset.tokenizerDelimiter).toBeInstanceOf(RegExp);
    });
});
