import { describe, it, expect, vi, beforeEach } from 'vitest';
import { IntentClassifier } from '../src/tasks/IntentClassifier';
import { ELM } from '../src/core/ELM';
import type { PredictResult } from '../src/core/ELM';

// Mock the ELM class
vi.mock('../src/core/ELM', () => {
    return {
        ELM: vi.fn().mockImplementation(() => ({
            train: vi.fn(),
            predict: vi.fn((text: string) => {
                return text.includes('hello')
                    ? [{ label: 'greeting', prob: 0.9 }, { label: 'goodbye', prob: 0.1 }]
                    : [{ label: 'goodbye', prob: 0.3 }];
            }),
        })),
    };
});

describe('IntentClassifier', () => {
    let classifier: IntentClassifier;

    beforeEach(() => {
        classifier = new IntentClassifier({ categories: [], hiddenUnits: 10, maxLen: 10, activation: 'relu' });
    });

    it('calls ELM.train and sets label set correctly', () => {
        const trainData = [
            { text: 'hi', label: 'greeting' },
            { text: 'bye', label: 'goodbye' },
        ];

        expect(() => classifier.train(trainData)).not.toThrow();
    });

    it('returns topK predictions above threshold', () => {
        const result = classifier.predict('hello', 2, 0.2);
        expect(result).toEqual([{ label: 'greeting', prob: 0.9 }]);
    });

    it('filters out predictions below threshold', () => {
        const result = classifier.predict('bye', 2, 0.5);
        expect(result).toEqual([]); // goodbye only has 0.3 prob
    });

    it('returns batch predictions', () => {
        const batch = classifier.predictBatch(['hello', 'bye'], 2, 0.2);
        expect(batch.length).toBe(2);
        expect(batch[0]).toEqual(
            expect.arrayContaining([{ label: 'greeting', prob: 0.9 }])
        );
        expect(batch[1]).toEqual(
            expect.arrayContaining([{ label: 'goodbye', prob: 0.3 }])
        );
    });

    it('does not throw when training with repeated labels', () => {
        const data = [
            { text: 'hi', label: 'greeting' },
            { text: 'hey', label: 'greeting' },
        ];
        expect(() => classifier.train(data)).not.toThrow();
    });
});
