import fs from "fs";
import { parse } from "csv-parse/sync";
import { pipeline } from "@xenova/transformers";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";

(async () => {
    const csvFile = fs.readFileSync("../public/ag-news-classification-dataset/train.csv", "utf8");
    const raw = parse(csvFile, { skip_empty_lines: true }) as string[][];
    const records = raw.map(row => ({ text: row[1].trim(), label: row[0].trim() }));
    const sampleSize = 1000;
    const texts = records.slice(0, sampleSize).map(r => r.text);
    const labels = records.slice(0, sampleSize).map(r => r.label);

    console.log(`⏳ Loading Sentence-BERT...`);
    const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    const bertTensor = await embedder(texts, { pooling: 'mean' });
    const bertEmbeddings = bertTensor.tolist() as number[][];

    const splitIdx = Math.floor(texts.length * 0.2);
    const queryLabels = labels.slice(0, splitIdx);
    const refLabels = labels.slice(splitIdx);
    const queryBERT = bertEmbeddings.slice(0, splitIdx);
    const refBERT = bertEmbeddings.slice(splitIdx);

    function cosineSimilarity(a: number[], b: number[]): number {
        const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
        const normA = Math.sqrt(a.reduce((s, ai) => s + ai * ai, 0));
        const normB = Math.sqrt(b.reduce((s, bi) => s + bi * bi, 0));
        return dot / (normA * normB);
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

    const bertResults = evaluateRecallMRR(queryBERT, refBERT, queryLabels, refLabels, 5);
    const targetRecall5 = bertResults.recallK;

    const activations = ["relu", "tanh"];
    const hiddenUnitOptions = [64, 128, 256];
    const maxLayers = 4;

    const csvLines = ["config,recall_at_1,recall_at_5,mrr"];
    let foundOptimal = false;

    for (let depth = 1; depth <= maxLayers; depth++) {
        for (const hidden of hiddenUnitOptions) {
            for (const act of activations) {
                const layers = Array.from({ length: depth }, () => ({ hiddenUnits: hidden, activation: act }));
                console.log(`\n🔹 Testing config: ${depth} layers, ${hidden} units, ${act}`);

                const elms = layers.map((cfg, i) => new ELM({ activation: cfg.activation, hiddenUnits: cfg.hiddenUnits, maxLen: 50, categories: [], log: { modelName: `ELM layer ${i + 1}`, verbose: false, toFile: false }, metrics: { accuracy: 0.01 } }));
                const chain = new ELMChain(elms);

                const encoder = elms[0].encoder;
                const encodedVectors = texts.map(t => encoder.normalize(encoder.encode(t)));
                let inputs = encodedVectors;
                for (const elm of elms) {
                    elm.trainFromData(inputs, inputs);
                    inputs = inputs.map(vec => elm.getEmbedding([vec])[0]);
                }

                const queryELM = inputs.slice(0, splitIdx);
                const refELM = inputs.slice(splitIdx);

                const elmResults = evaluateRecallMRR(queryELM, refELM, queryLabels, refLabels, 5);

                csvLines.push(`ELMChain_${depth}_${hidden}_${act},${elmResults.recall1.toFixed(4)},${elmResults.recallK.toFixed(4)},${elmResults.mrr.toFixed(4)}`);

                if (elmResults.recallK >= targetRecall5) {
                    console.log(`✅ Found optimal config matching BERT Recall@5: ${elmResults.recallK.toFixed(4)}`);
                    foundOptimal = true;
                    break;
                }
            }
            if (foundOptimal) break;
        }
        if (foundOptimal) break;
    }

    // Add BERT baseline result to the CSV header
    csvLines.unshift(`SentenceBERT_Baseline,${bertResults.recall1.toFixed(4)},${bertResults.recallK.toFixed(4)},${bertResults.mrr.toFixed(4)}`);

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `automated_experiment_results_${timestamp}.csv`;
    fs.writeFileSync(filename, csvLines.join("\n"));
    console.log(`\n✅ Experiment complete. Results saved to ${filename}.`);
})();
