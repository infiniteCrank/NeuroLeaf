import fs from "fs";
import { parse } from "csv-parse/sync";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";
import { TFIDF } from "../src/core/TFIDF";

(async () => {
    const csvFile = fs.readFileSync("../public/ag-news-classification-dataset/train.csv", "utf8");
    const raw = parse(csvFile, { skip_empty_lines: true }) as string[][];
    const records = raw.map(row => ({ text: row[1].trim(), label: row[0].trim() }));
    const sampleSize = 1000; // reduced sample size
    const texts = records.slice(0, sampleSize).map(r => r.text);
    const labels = records.slice(0, sampleSize).map(r => r.label);

    const splitIdx = Math.floor(texts.length * 0.2);
    const queryLabels = labels.slice(0, splitIdx);
    const refLabels = labels.slice(splitIdx);

    const tfidfModel = new TFIDF(texts);
    tfidfModel.calculateScores();

    const vocab = Object.keys(tfidfModel.scores).slice(0, 1000); // limit vocab size
    const tfidfEmbeddings = texts.map(t => {
        const tokens = t.split(/\s+/);
        const processed = TFIDF.processWords(tokens);
        return vocab.map(token => processed.includes(token) ? tfidfModel.scores[token] || 0 : 0);
    });

    const hiddenUnits = [256, 128]; // reduced model size
    const activations = ["relu", "tanh"];
    const dropout = 0.02;

    const elms = hiddenUnits.map((h, i) => new ELM({
        activation: activations[i],
        hiddenUnits: h,
        maxLen: 50,
        categories: [],
        log: { modelName: `ELM layer ${i + 1}`, verbose: false, toFile: false },
        metrics: { accuracy: 0.01 },
        dropout
    }));

    const encoder = elms[0].encoder;
    const encodedVectors = texts.map(t => encoder.normalize(encoder.encode(t)));

    const batchSize = 200;
    let inputs: number[][] = [];

    // Process batches to avoid memory issues
    for (let start = 0; start < encodedVectors.length; start += batchSize) {
        const batch = encodedVectors.slice(start, start + batchSize);
        let batchInputs = batch;

        for (const elm of elms) {
            const targetDim = Math.min(batchInputs[0].length, elm.hiddenUnits);
            const randomTargetMatrix = Array.from({ length: batchInputs.length }, () => Array.from({ length: targetDim }, () => Math.random()));
            elm.trainFromData(batchInputs, randomTargetMatrix, { reuseWeights: false });
            batchInputs = batchInputs.map(vec => elm.getEmbedding([vec])[0]);
        }

        inputs.push(...batchInputs);
    }

    const embeddingStore: EmbeddingRecord[] = records.slice(0, sampleSize).map((rec, i) => ({
        embedding: l2normalize(inputs[i]),
        metadata: { text: rec.text, label: rec.label }
    }));

    fs.writeFileSync(`elm_embeddings.json`, JSON.stringify(embeddingStore, null, 2));

    const queries = embeddingStore.slice(0, splitIdx);
    const reference = embeddingStore.slice(splitIdx);

    const results = evaluateRecallMRR(
        queries.map(q => q.embedding),
        reference.map(r => r.embedding),
        queryLabels,
        refLabels,
        5
    );

    console.log(`Recall@1: ${(results.recall1 * 100).toFixed(2)}%`);
    console.log(`Recall@5: ${(results.recallK * 100).toFixed(2)}%`);
    console.log(`MRR: ${results.mrr.toFixed(4)}`);

    function cosineSimilarity(a: number[], b: number[]): number {
        const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
        const normA = Math.sqrt(a.reduce((s, ai) => s + ai * ai, 0));
        const normB = Math.sqrt(b.reduce((s, bi) => s + bi * bi, 0));
        return normA && normB ? dot / (normA * normB) : 0;
    }

    function l2normalize(v: number[]): number[] {
        const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
        return norm === 0 ? v : v.map(x => x / norm);
    }

    function evaluateRecallMRR(query: number[][], reference: number[][], qLabels: string[], rLabels: string[], k: number) {
        let hitsAt1 = 0, hitsAtK = 0, reciprocalRanks = 0;
        for (let i = 0; i < query.length; i++) {
            const scores = reference.map((emb, j) => ({ label: rLabels[j], score: cosineSimilarity(query[i], emb) }));
            scores.sort((a, b) => b.score - a.score);
            const ranked = scores.map(s => s.label);
            if (ranked[0] === qLabels[i]) hitsAt1++;
            if (ranked.slice(0, k).includes(qLabels[i])) hitsAtK++;
            const rank = ranked.indexOf(qLabels[i]);
            reciprocalRanks += rank === -1 ? 0 : 1 / (rank + 1);
        }
        return { recall1: hitsAt1 / query.length, recallK: hitsAtK / query.length, mrr: reciprocalRanks / query.length };
    }
})();
