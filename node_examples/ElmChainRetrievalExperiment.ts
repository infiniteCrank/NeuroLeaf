import fs from 'fs';
import { parse } from 'csv-parse/sync';
import { ELM } from '../src/core/ELM';
import { ELMChain } from '../src/core/ELMChain';
import { EmbeddingRecord } from '../src/core/EmbeddingStore';
import { evaluateRetrieval } from '../src/core/Evaluation';

// Load AG News data (train.csv)
const csvFile = fs.readFileSync('./train.csv', 'utf8');
const raw = parse(csvFile, {
    skip_empty_lines: true
}) as string[][];

// AG News columns: [label, title, description]
const records = raw.map(row => ({
    text: row[1].trim().toLowerCase(),
    label: row[0].trim()
}));

// Limit for faster experimentation
const sampleSize = 1000;
const sampleRecords = records.slice(0, sampleSize);

// Prepare inputs
const texts = sampleRecords.map(r => r.text);
const labels = sampleRecords.map(r => r.label);

// Initialize two ELMs
const elm1 = new ELM({
    activation: 'relu',
    hiddenUnits: 128,
    maxLen: 50,
    categories: [],
    log: {
        modelName: "Elm link 1",
        verbose: true,
        toFile: false
    }
});

const elm2 = new ELM({
    activation: 'tanh',
    hiddenUnits: 64,
    maxLen: 128,
    categories: [],
    log: {
        modelName: "Elm link 2",
        verbose: true,
        toFile: false
    }
});

// Create ELMChain
const chain = new ELMChain([elm1, elm2]);

// Encode inputs using UniversalEncoder
const encoder = elm1.encoder;
const encodedVectors = texts.map(t => encoder.normalize(encoder.encode(t)));

// Train first ELM to reproduce inputs
elm1.trainFromData(encodedVectors, encodedVectors);
const elm1Outputs = encodedVectors.map(vec => elm1.getEmbedding([vec])[0]);

// Train second ELM to reproduce outputs of the first
elm2.trainFromData(elm1Outputs, elm1Outputs);

// Build embedding store
const embeddingStore: EmbeddingRecord[] = sampleRecords.map((rec, i) => ({
    embedding: chain.getEmbedding([encodedVectors[i]])[0],
    metadata: { text: rec.text, label: rec.label }
}));

// Split into queries and reference set
const splitIdx = Math.floor(embeddingStore.length * 0.2);
const queries = embeddingStore.slice(0, splitIdx);
const reference = embeddingStore.slice(splitIdx);

// Evaluate retrieval performance
const results = evaluateRetrieval(queries, reference, chain, 5);

console.log(`\n=== Retrieval Evaluation Results ===`);
console.log(`Recall@5: ${results.recallAtK.toFixed(4)}`);
console.log(`MRR: ${results.mrr.toFixed(4)}`);

// Save results
fs.writeFileSync(
    'experiment_results.csv',
    `config,recall_at_5,mrr\nELMChain_2_layers,${results.recallAtK.toFixed(4)},${results.mrr.toFixed(4)}\n`
);

console.log(`✅ Results saved to experiment_results.csv.`);
