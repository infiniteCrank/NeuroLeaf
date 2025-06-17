import { describe, it, expect, vi, beforeEach } from 'vitest';
import { LanguageClassifier } from '../src/tasks/LanguageClassifier';
import { ELM } from '../src/core/ELM';
import { IO } from '../src/utils/IO';

vi.mock('../src/core/ELM', () => ({
    ELM: vi.fn().mockImplementation(() => ({
        setCategories: vi.fn(),
        train: vi.fn(),
        predict: vi.fn().mockReturnValue([
            { label: 'english', prob: 0.95 },
            { label: 'spanish', prob: 0.03 }
        ])
    }))
}));

vi.mock('../src/utils/IO', () => ({
    IO: {
        importJSON: vi.fn(() => [{ text: 'hello', label: 'english' }]),
        importCSV: vi.fn(() => [{ text: 'hola', label: 'spanish' }]),
        importTSV: vi.fn(() => [{ text: 'bonjour', label: 'french' }])
    }
}));

describe('LanguageClassifier', () => {
    let classifier: LanguageClassifier;

    beforeEach(() => {
        classifier = new LanguageClassifier({
            categories: [],
            hiddenUnits: 10,
            maxLen: 10,
            activation: 'relu'
        });
    });

    it('loads training data from JSON', () => {
        const raw = '[{"text":"hello","label":"english"}]';
        const data = classifier.loadTrainingData(raw, 'json');
        expect(IO.importJSON).toHaveBeenCalledWith(raw);
        expect(data).toEqual([{ text: 'hello', label: 'english' }]);
    });

    it('loads training data from CSV', () => {
        const raw = 'text,label\nhola,spanish';
        const data = classifier.loadTrainingData(raw, 'csv');
        expect(IO.importCSV).toHaveBeenCalledWith(raw);
        expect(data[0].label).toBe('spanish');
    });

    it('loads training data from TSV', () => {
        const raw = 'text\tlabel\nbonjour\tfrench';
        const data = classifier.loadTrainingData(raw, 'tsv');
        expect(IO.importTSV).toHaveBeenCalledWith(raw);
        expect(data[0].label).toBe('french');
    });

    it('trains with deduplicated categories and builds trainSamples', () => {
        const data = [
            { text: 'hi', label: 'english' },
            { text: 'hello', label: 'english' },
            { text: 'hola', label: 'spanish' }
        ];
        classifier.train(data);

        const elm = (classifier as any).elm;
        expect(elm.setCategories).toHaveBeenCalledWith(['english', 'spanish']);
        expect(elm.train).toHaveBeenCalled();

        const samples = (classifier as any).trainSamples;
        expect(samples['english']).toEqual(['hi', 'hello']);
        expect(samples['spanish']).toEqual(['hola']);
    });

    it('predicts using ELM.predict', () => {
        const result = classifier.predict('hello');
        expect(result).toEqual([
            { label: 'english', prob: 0.95 },
            { label: 'spanish', prob: 0.03 }
        ]);
    });
});
