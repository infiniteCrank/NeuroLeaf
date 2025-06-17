// IO.test.ts - Unit tests for IO utilities

import { describe, it, expect } from 'vitest';
import { IO } from '../src/utils/IO';

const sampleJSON = `[
  {"text": "hello", "label": "greeting"},
  {"text": "bye", "label": "farewell"}
]`;

const sampleCSV = `text,label\nhello,greeting\nbye,farewell`;
const sampleTSV = `text\tlabel\nhello\tgreeting\nbye\tfarewell`;

describe('IO', () => {
    it('imports JSON correctly', () => {
        const data = IO.importJSON(sampleJSON);
        expect(data).toEqual([
            { text: 'hello', label: 'greeting' },
            { text: 'bye', label: 'farewell' },
        ]);
    });

    it('exports JSON correctly', () => {
        const json = IO.exportJSON([
            { text: 'hi', label: 'greet' },
        ]);
        expect(json).toContain('hi');
        expect(json).toContain('greet');
    });

    it('imports CSV correctly', () => {
        const data = IO.importCSV(sampleCSV);
        expect(data.length).toBe(2);
        expect(data[0].text).toBe('hello');
    });

    it('exports CSV correctly', () => {
        const csv = IO.exportCSV([
            { text: 'yo', label: 'greet' },
        ]);
        expect(csv).toContain('yo,greet');
    });

    it('imports TSV correctly', () => {
        const data = IO.importTSV(sampleTSV);
        expect(data.length).toBe(2);
        expect(data[1].label).toBe('farewell');
    });

    it('exports TSV correctly', () => {
        const tsv = IO.exportTSV([
            { text: 'hey', label: 'greeting' },
        ]);
        expect(tsv).toContain('hey\tgreeting');
    });

    it('infers schema from CSV correctly', () => {
        const schema = IO.inferSchemaFromCSV(sampleCSV);
        expect(schema.fields).toEqual([
            { name: 'text', type: 'string' },
            { name: 'label', type: 'string' },
        ]);
        expect(schema.suggestedMapping?.text).toBe('text');
        expect(schema.suggestedMapping?.label).toBe('label');
    });

    it('infers schema from JSON correctly', () => {
        const schema = IO.inferSchemaFromJSON(sampleJSON);
        expect(schema.fields).toEqual([
            { name: 'text', type: 'string' },
            { name: 'label', type: 'string' },
        ]);
        expect(schema.suggestedMapping?.text).toBe('text');
    });

    it('returns empty array on invalid JSON import', () => {
        const data = IO.importJSON('{invalid json');
        expect(data).toEqual([]);
    });
});
