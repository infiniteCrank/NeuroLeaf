import fs from 'fs';
import { parse } from 'csv-parse/sync';
import { ELM } from '../src/core/ELM';
import { ELMChain } from '../src/core/ELMChain';
import { EmbeddingRecord } from '../src/core/EmbeddingStore';
import { evaluateRetrieval } from '../src/core/Evaluation';

// Load AG News data
const csvFile = fs.readFileSync('../public/ag-news-classification-dataset/train.csv', 'utf8');
const raw = parse(csvFile, { skip_empty_lines: true }) as string[][];

const records = raw.map(row => ({
    text: row[1].trim().toLowerCase(),
    label: row[0].trim()
}));

// Sample smaller subset for quick testing
const sampleSize = 1000;
const sampleRecords = records.slice(0, sampleSize);

const texts = sampleRecords.map(r => r.text);
const labels = sampleRecords.map(r => r.label);

interface LayerConfig {
    hiddenUnits: number;
    activation: string;
}

interface ExperimentConfig {
    name: string;
    layers: LayerConfig[];
}

const experiments: ExperimentConfig[] = [
    {
        name: "ELMChain_2_layers_relu_tanh",
        layers: [
            { hiddenUnits: 128, activation: 'relu' },
            { hiddenUnits: 64, activation: 'tanh' }
        ]
    },
    {
        name: "ELMChain_2_layers_relu_relu",
        layers: [
            { hiddenUnits: 128, activation: 'relu' },
            { hiddenUnits: 64, activation: 'relu' }
        ]
    },
    {
        name: "ELMChain_3_layers_deep",
        layers: [
            { hiddenUnits: 256, activation: 'relu' },
            { hiddenUnits: 128, activation: 'relu' },
            { hiddenUnits: 64, activation: 'tanh' }
        ]
    }
];

(async () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const resultsFilename = `experiment_results_${timestamp}.csv`;
    const csvHeader = "config,start_time,end_time,recall_at_5,mrr\n";
    fs.writeFileSync(resultsFilename, csvHeader);

    for (const exp of experiments) {
        console.log(`\n🚀 Running experiment: ${exp.name}`);
        const startTime = new Date().toISOString();

        const elms: ELM[] = exp.layers.map((layer, i) => new ELM({
            activation: layer.activation,
            hiddenUnits: layer.hiddenUnits,
            maxLen: 50,
            categories: [],
            log: {
                modelName: `ELM layer ${i + 1}`,
                verbose: true,
                toFile: false
            },
            metrics: { accuracy: 0.01 }
        }));

        const chain = new ELMChain(elms);

        const encoder = elms[0].encoder;
        const encodedVectors = texts.map(t => encoder.normalize(encoder.encode(t)));

        let inputs = encodedVectors;
        for (let i = 0; i < elms.length; i++) {
            console.log(`⚙️ Training ELM ${i + 1} (${exp.layers[i].activation}, ${exp.layers[i].hiddenUnits})`);
            elms[i].trainFromData(inputs, inputs);
            inputs = inputs.map(vec => elms[i].getEmbedding([vec])[0]);
        }

        const embeddingStore: EmbeddingRecord[] = sampleRecords.map((rec, i) => ({
            embedding: chain.getEmbedding([encodedVectors[i]])[0],
            metadata: { text: rec.text, label: rec.label }
        }));

        const splitIdx = Math.floor(embeddingStore.length * 0.2);
        const queries = embeddingStore.slice(0, splitIdx);
        const reference = embeddingStore.slice(splitIdx);

        const results = evaluateRetrieval(queries, reference, chain, 5);
        const endTime = new Date().toISOString();

        console.log(`✅ Results for ${exp.name}`);
        console.log(`Recall@5: ${results.recallAtK.toFixed(4)}`);
        console.log(`MRR: ${results.mrr.toFixed(4)}`);

        const csvLine = `${exp.name},${startTime},${endTime},${results.recallAtK.toFixed(4)},${results.mrr.toFixed(4)}\n`;
        fs.appendFileSync(resultsFilename, csvLine);
    }

    console.log(`\n🎉 All experiments complete. Results saved to ${resultsFilename}`);
})();
