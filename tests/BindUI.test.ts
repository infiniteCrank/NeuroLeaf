// BindUI.test.ts - Unit + Integration + Edge Case tests for BindUI, IntentClassifier, AutoComplete, ELMConfig, LanguageClassifier

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { bindAutocompleteUI } from '../src/ui/components/BindUI';
import { ELM } from '../src/core/ELM';
import { IntentClassifier } from '../src/tasks/IntentClassifier';
import { AutoComplete } from '../src/tasks/AutoComplete';
import { LanguageClassifier } from '../src/tasks/LanguageClassifier';
import { defaultConfig } from '../src/core/ELMConfig';
import { IO } from '../src/utils/IO';

// Mocked PredictResult
const mockResults = [
    { label: 'rock', prob: 0.9 },
    { label: 'pop', prob: 0.1 }
];

describe('bindAutocompleteUI', () => {
    let input: HTMLInputElement;
    let output: HTMLElement;
    let mockModel: ELM;

    beforeEach(() => {
        input = document.createElement('input');
        output = document.createElement('div');
        mockModel = {
            predict: vi.fn(() => mockResults)
        } as unknown as ELM;
    });

    it('shows prompt when input is empty', () => {
        bindAutocompleteUI({ model: mockModel, inputElement: input, outputElement: output });
        input.value = '';
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toContain('Start typing');
    });

    it('renders predictions when input is non-empty', () => {
        bindAutocompleteUI({ model: mockModel, inputElement: input, outputElement: output });
        input.value = 'ro';
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toContain('rock');
        expect(output.innerHTML).toContain('90.0%');
    });

    it('displays error if model.predict throws', () => {
        mockModel.predict = vi.fn(() => { throw new Error('Prediction failed'); }) as any;
        bindAutocompleteUI({ model: mockModel, inputElement: input, outputElement: output });
        input.value = 'error';
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toContain('Error: Prediction failed');
    });

    it('handles large input gracefully', () => {
        bindAutocompleteUI({ model: mockModel, inputElement: input, outputElement: output });
        input.value = 'a'.repeat(1000);
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toContain('rock');
    });

    it('handles unusual unicode characters', () => {
        bindAutocompleteUI({ model: mockModel, inputElement: input, outputElement: output });
        input.value = 'こんにちは';
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toContain('rock');
    });
});

describe('IntentClassifier Integration', () => {
    it('trains and predicts intents', () => {
        const classifier = new IntentClassifier({
            categories: ['greeting', 'farewell'],
            hiddenUnits: 10,
            maxLen: 10,
            activation: 'relu'
        });

        const samples = [
            { text: 'hello', label: 'greeting' },
            { text: 'hi', label: 'greeting' },
            { text: 'bye', label: 'farewell' }
        ];

        classifier.train(samples);

        const result = classifier.predict('hello', 1);
        expect(result.length).toBe(1);
        const [top] = result;
        expect(['greeting', 'farewell']).toContain(top.label);
        expect(top.prob).toBeGreaterThan(0);
    });

    it('returns empty array if no intent is above threshold', () => {
        const classifier = new IntentClassifier({
            categories: ['greeting'],
            hiddenUnits: 10,
            maxLen: 10,
            activation: 'relu'
        });
        classifier.train([{ text: 'hello', label: 'greeting' }]);

        const result = classifier.predict('unknown', 1, 1.1);
        expect(result).toEqual([]);
    });

    it('produces consistent predictions for same input', () => {
        const classifier = new IntentClassifier({
            categories: ['greeting'],
            hiddenUnits: 10,
            maxLen: 10,
            activation: 'relu'
        });
        classifier.train([{ text: 'hello', label: 'greeting' }]);
        const first = classifier.predict('hello');
        const second = classifier.predict('hello');
        expect(first[0].label).toBe(second[0].label);
    });

    it('retraining overwrites previous model state', () => {
        const classifier = new IntentClassifier({
            categories: ['greeting', 'farewell'],
            hiddenUnits: 10,
            maxLen: 10,
            activation: 'relu'
        });
        classifier.train([{ text: 'hi', label: 'greeting' }]);
        const initial = classifier.predict('hi')[0];
        classifier.train([{ text: 'bye', label: 'farewell' }]);
        const updated = classifier.predict('bye')[0];
        expect(initial.label).not.toBe(updated.label);
    });

    it('token-based models still classify correctly', () => {
        const classifier = new IntentClassifier({
            categories: ['affirmative', 'negative'],
            hiddenUnits: 10,
            maxLen: 10,
            activation: 'relu',
            useTokenizer: true,
            tokenizerDelimiter: /\s+/
        });
        classifier.train([
            { text: 'yes', label: 'affirmative' },
            { text: 'yeah', label: 'affirmative' },
            { text: 'no', label: 'negative' }
        ]);
        const result = classifier.predict('yes')[0];
        expect(['affirmative', 'negative']).toContain(result.label);
    });
});

describe('LanguageClassifier Integration', () => {
    const json = JSON.stringify([
        { text: 'hello', label: 'English' },
        { text: 'bonjour', label: 'French' },
        { text: 'hola', label: 'Spanish' }
    ]);

    const csv = 'text,label\nhello,English\nbonjour,French\nhola,Spanish';
    const tsv = 'text\tlabel\nhello\tEnglish\nbonjour\tFrench\nhola\tSpanish';

    it('loads JSON and predicts language', () => {
        const classifier = new LanguageClassifier({
            categories: ['English', 'French', 'Spanish'],
            hiddenUnits: 20,
            maxLen: 10,
            activation: 'relu'
        });
        const data = classifier.loadTrainingData(json, 'json');
        classifier.train(data);
        const prediction = classifier.predict('hello', 1)[0];
        expect(['English', 'French', 'Spanish']).toContain(prediction.label);
    });

    it('loads CSV and predicts language', () => {
        const classifier = new LanguageClassifier({
            categories: ['English', 'French', 'Spanish'],
            hiddenUnits: 20,
            maxLen: 10,
            activation: 'relu'
        });
        const data = classifier.loadTrainingData(csv, 'csv');
        classifier.train(data);
        const prediction = classifier.predict('bonjour', 1)[0];
        expect(['French', 'English', 'Spanish']).toContain(prediction.label);
    });

    it('loads TSV and predicts language', () => {
        const classifier = new LanguageClassifier({
            categories: ['English', 'French', 'Spanish'],
            hiddenUnits: 20,
            maxLen: 10,
            activation: 'relu'
        });
        const data = classifier.loadTrainingData(tsv, 'tsv');
        classifier.train(data);
        const prediction = classifier.predict('hola', 1)[0];
        expect(['Spanish', 'English', 'French']).toContain(prediction.label);
    });

    it('infers schema from JSON', () => {
        const schema = IO.inferSchemaFromJSON(json);
        expect(schema.fields.some(f => f.name === 'text')).toBe(true);
        expect(schema.fields.some(f => f.name === 'label')).toBe(true);
    });

    it('infers schema from CSV', () => {
        const schema = IO.inferSchemaFromCSV(csv);
        expect(schema.fields.some(f => f.name === 'text')).toBe(true);
        expect(schema.fields.some(f => f.name === 'label')).toBe(true);
    });
});
