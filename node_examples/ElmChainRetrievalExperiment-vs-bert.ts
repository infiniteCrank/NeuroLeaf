import fs from 'fs';
import { parse } from 'csv-parse/sync';
import { pipeline } from '@xenova/transformers';
import { ELM } from '../src/core/ELM';
import { ELMChain } from '../src/core/ELMChain';
import { PCA } from 'ml-pca';

// Load AG News
const csvFile = fs.readFileSync('../public/ag-news-classification-dataset/train.csv', 'utf8');
const raw = parse(csvFile, { skip_empty_lines: true }) as string[][];

const records = raw.map(row => ({
    text: row[1].trim(),
    label: row[0].trim()
}));

const sampleSize = 1000;
const sampleRecords = records.slice(0, sampleSize);

const texts = sampleRecords.map(r => r.text);
const labels = sampleRecords.map(r => r.label);

(async () => {
    console.log(`⏳ Loading Sentence-BERT model...`);
    const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log(`✅ Sentence-BERT ready.`);

    console.log(`⏳ Generating Sentence-BERT embeddings...`);
    const bertTensor = await embedder(texts, { pooling: 'mean' });
    const bertEmbeddings = bertTensor.tolist() as number[][];
    console.log(`✅ Sentence-BERT embeddings done.`);

    console.log(`⏳ Training ELMChain...`);
    const elm1 = new ELM({
        activation: 'relu',
        hiddenUnits: 128,
        maxLen: 50,
        categories: [],
        log: { modelName: 'ELM layer 1', verbose: true, toFile: false },
        metrics: { accuracy: 0.01 }
    });
    const elm2 = new ELM({
        activation: 'tanh',
        hiddenUnits: 64,
        maxLen: 128,
        categories: [],
        log: { modelName: 'ELM layer 2', verbose: true, toFile: false },
        metrics: { accuracy: 0.01 }
    });

    const chain = new ELMChain([elm1, elm2]);
    const encoder = elm1.encoder;
    const encodedVectors = texts.map(t => encoder.normalize(encoder.encode(t)));

    elm1.trainFromData(encodedVectors, encodedVectors);
    const elm1Outputs = encodedVectors.map(vec => elm1.getEmbedding([vec])[0]);
    elm2.trainFromData(elm1Outputs, elm1Outputs);

    const elmEmbeddings = encodedVectors.map(vec => chain.getEmbedding([vec])[0]);

    // Split queries/reference
    const splitIdx = Math.floor(sampleRecords.length * 0.2);

    const queryELM = elmEmbeddings.slice(0, splitIdx);
    const refELM = elmEmbeddings.slice(splitIdx);

    const queryBERT = bertEmbeddings.slice(0, splitIdx);
    const refBERT = bertEmbeddings.slice(splitIdx);

    const queryLabels = labels.slice(0, splitIdx);
    const refLabels = labels.slice(splitIdx);

    function cosineSimilarity(a: number[], b: number[]): number {
        const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
        const normA = Math.sqrt(a.reduce((s, ai) => s + ai * ai, 0));
        const normB = Math.sqrt(b.reduce((s, bi) => s + bi * bi, 0));
        return dot / (normA * normB);
    }

    function evaluateRecallMRR(query: number[][], reference: number[][], queryLabels: string[], refLabels: string[], k: number) {
        let hitsAt1 = 0;
        let hitsAtK = 0;
        let reciprocalRanks = 0;

        for (let i = 0; i < query.length; i++) {
            const scores = reference.map((emb, j) => ({
                label: refLabels[j],
                score: cosineSimilarity(query[i], emb)
            }));
            scores.sort((a, b) => b.score - a.score);
            const rankedLabels = scores.map(s => s.label);

            if (rankedLabels[0] === queryLabels[i]) hitsAt1++;
            if (rankedLabels.slice(0, k).includes(queryLabels[i])) hitsAtK++;
            const rank = rankedLabels.indexOf(queryLabels[i]);
            reciprocalRanks += rank === -1 ? 0 : 1 / (rank + 1);
        }

        return {
            recall1: hitsAt1 / query.length,
            recallK: hitsAtK / query.length,
            mrr: reciprocalRanks / query.length
        };
    }

    const elmResults = evaluateRecallMRR(queryELM, refELM, queryLabels, refLabels, 5);
    const bertResults = evaluateRecallMRR(queryBERT, refBERT, queryLabels, refLabels, 5);

    console.log(`\n=== ELMChain Results ===`);
    console.log(`Recall@1: ${elmResults.recall1.toFixed(4)}`);
    console.log(`Recall@5: ${elmResults.recallK.toFixed(4)}`);
    console.log(`MRR: ${elmResults.mrr.toFixed(4)}`);

    console.log(`\n=== Sentence-BERT Results ===`);
    console.log(`Recall@1: ${bertResults.recall1.toFixed(4)}`);
    console.log(`Recall@5: ${bertResults.recallK.toFixed(4)}`);
    console.log(`MRR: ${bertResults.mrr.toFixed(4)}`);

    fs.writeFileSync(
        'comparison_results.csv',
        `model,recall_at_1,recall_at_5,mrr\n` +
        `ELMChain,${elmResults.recall1.toFixed(4)},${elmResults.recallK.toFixed(4)},${elmResults.mrr.toFixed(4)}\n` +
        `SentenceBERT,${bertResults.recall1.toFixed(4)},${bertResults.recallK.toFixed(4)},${bertResults.mrr.toFixed(4)}\n`
    );

    // PCA visualization export
    function runPCAAndSave(embeddings: number[][], labels: string[], model: string) {
        const pca = new PCA(embeddings);
        const reduced = pca.predict(embeddings, { nComponents: 2 }).to2DArray();
        const output = reduced.map((point, i) => ({
            x: point[0],
            y: point[1],
            label: labels[i]
        }));
        fs.writeFileSync(`${model}_pca_2d.json`, JSON.stringify(output, null, 2));
        console.log(`✅ PCA for ${model} saved to ${model}_pca_2d.json`);
    }

    runPCAAndSave(elmEmbeddings, labels, 'ELMChain');
    runPCAAndSave(bertEmbeddings, labels, 'SentenceBERT');

    console.log(`\n✅ Results and PCA files generated.`);
})();
