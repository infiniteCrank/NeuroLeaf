import fs from "fs";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";

function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 ? v : v.map(x => x / norm);
}

function zeroCenter(vectors: number[][]): number[][] {
    const mean = vectors[0].map((_, j) =>
        vectors.reduce((s, v) => s + v[j], 0) / vectors.length
    );
    return vectors.map(v =>
        v.map((x, j) => x - mean[j])
    );
}

function averageVectors(vectors: number[][]): number[] {
    return vectors[0].map((_, i) =>
        vectors.reduce((s, v) => s + v[i], 0) / vectors.length
    );
}

// 1️⃣ Load corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");

// 2️⃣ Split into paragraphs
const paragraphs = rawText
    .split(/\n{2,}/)
    .map(p => p.trim())
    .filter(p => p && p.length > 30);

console.log(`✅ Parsed ${paragraphs.length} paragraphs.`);

// 3️⃣ Create UniversalEncoder
const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyz0123456789",
    mode: "token",
    useTokenizer: true
});

// Word-level embeddings
const wordVectors = paragraphs.map(p => {
    const tokens = p.split(/\s+/).filter(Boolean);
    return l2normalize(averageVectors(tokens.map(t => encoder.normalize(encoder.encode(t)))));
});

// Sentence-level embeddings
const sentenceVectors = paragraphs.map(p => {
    const sentences = p.split(/[.?!]\s+/).filter(s => s.length > 3);
    return l2normalize(averageVectors(sentences.map(s => encoder.normalize(encoder.encode(s)))));
});

// Paragraph-level embeddings
const paragraphVectors = paragraphs.map(p =>
    l2normalize(encoder.normalize(encoder.encode(p)))
);

console.log(`✅ Prepared all input embeddings.`);

// 4️⃣ Train/load helper
function trainOrLoadELM(name: string, inputDim: number, vectors: number[][], hiddenUnits: number) {
    const elm = new ELM({
        activation: "relu",
        hiddenUnits,
        maxLen: inputDim,
        categories: [],
        log: { modelName: name, verbose: true },
        dropout: 0.02
    });

    const weightsPath = `./elm_weights/${name}.json`;
    if (fs.existsSync(weightsPath)) {
        const saved = fs.readFileSync(weightsPath, "utf-8");
        elm.loadModelFromJSON(saved);
        console.log(`✅ Loaded weights for ${name}.`);
    } else {
        console.log(`⚙️ Training ${name}...`);
        elm.trainFromData(vectors, vectors);
        if (elm.model) {
            fs.writeFileSync(weightsPath, JSON.stringify(elm.model));
            console.log(`💾 Saved weights for ${name}.`);
        }
    }
    return elm;
}

// 5️⃣ Train/load ELMs
const wordELM = trainOrLoadELM("word_encoder", wordVectors[0].length, wordVectors, 64);
const sentenceELM = trainOrLoadELM("sentence_encoder", sentenceVectors[0].length, sentenceVectors, 64);
const paragraphELM = trainOrLoadELM("paragraph_encoder", paragraphVectors[0].length, paragraphVectors, 128);

// 6️⃣ Compute embeddings
const wordEmb = wordELM.computeHiddenLayer(wordVectors);
const sentenceEmb = sentenceELM.computeHiddenLayer(sentenceVectors);
const paragraphEmb = paragraphELM.computeHiddenLayer(paragraphVectors);

// 7️⃣ Zero-center then L2-normalize each embedding
function processEmbeddings(embs: number[][]) {
    const centered = zeroCenter(embs);
    return centered.map(l2normalize);
}
const wordProcessed = processEmbeddings(wordEmb);
const sentProcessed = processEmbeddings(sentenceEmb);
const paraProcessed = processEmbeddings(paragraphEmb);

// 8️⃣ Concatenate
const combinedEmbeddings = wordProcessed.map((_, i) =>
    l2normalize([
        ...wordProcessed[i],
        ...sentProcessed[i],
        ...paraProcessed[i]
    ])
);

console.log(`✅ Combined embeddings.`);

// 9️⃣ Train indexer chain
const hiddenUnitSequence = [256, 128];
let embeddings = combinedEmbeddings;

const indexerELMs = hiddenUnitSequence.map((h, i) =>
    new ELM({
        activation: "relu",
        hiddenUnits: h,
        maxLen: embeddings[0].length,
        categories: [],
        log: { modelName: `IndexerELM_${i}`, verbose: true },
        dropout: 0.1 // 💡 Slightly higher dropout to improve variance
    })
);

indexerELMs.forEach((elm, i) => {
    const weightsPath = `./elm_weights/indexer_layer${i}.json`;
    if (fs.existsSync(weightsPath)) {
        const saved = fs.readFileSync(weightsPath, "utf-8");
        elm.loadModelFromJSON(saved);
        console.log(`✅ Loaded Indexer layer ${i}.`);
    } else {
        console.log(`⚙️ Training Indexer layer ${i}...`);
        elm.trainFromData(embeddings, embeddings);
        if (elm.model) {
            fs.writeFileSync(weightsPath, JSON.stringify(elm.model));
            console.log(`💾 Saved Indexer layer ${i}.`);
        }
    }
    embeddings = processEmbeddings(elm.computeHiddenLayer(embeddings));
});

const indexerChain = new ELMChain(indexerELMs);
console.log(`✅ Indexer ELM chain ready.`);

// 🔟 Save final embeddings
const embeddingRecords: EmbeddingRecord[] = paragraphs.map((p, i) => ({
    embedding: embeddings[i],
    metadata: { text: p }
}));
fs.mkdirSync("./embeddings", { recursive: true });
fs.writeFileSync("./embeddings/combined_embeddings.json", JSON.stringify(embeddingRecords, null, 2));
console.log(`💾 Saved combined embeddings.`);

// 🔍 Retrieval
function retrieve(query: string, topK = 5) {
    const tokens = query.split(/\s+/).filter(Boolean);
    const avgWord = l2normalize(averageVectors(tokens.map(t => encoder.normalize(encoder.encode(t)))));
    const sentVec = l2normalize(encoder.normalize(encoder.encode(query)));
    const paraVec = sentVec;

    const wordE = wordELM.computeHiddenLayer([avgWord])[0];
    const sentE = sentenceELM.computeHiddenLayer([sentVec])[0];
    const paraE = paragraphELM.computeHiddenLayer([paraVec])[0];

    const combined = l2normalize([
        ...wordE,
        ...sentE,
        ...paraE
    ]);

    const finalVec = indexerChain.getEmbedding([combined])[0];

    const scored = embeddingRecords.map(r => ({
        text: r.metadata.text,
        score: r.embedding.reduce((s, v, i) => s + v * finalVec[i], 0)
    }));

    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

const results = retrieve("How do you declare a map in Go?");
console.log(`\n🔍 Retrieval results:`);
results.forEach((r, i) =>
    console.log(` ${i + 1}. (${r.score.toFixed(4)}) ${r.text.slice(0, 80)}...`)
);

console.log(`\n✅ Done.`);
