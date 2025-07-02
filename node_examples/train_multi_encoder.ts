import fs from "fs";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";

// Utility: L2 normalize a vector
function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 ? v : v.map(x => x / norm);
}

// 1️⃣ Load corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");

// 2️⃣ Split into paragraphs
const paragraphs = rawText
    .split(/\n{2,}/)
    .map(p => p.trim())
    .filter(p => p && p.length > 30);

console.log(`✅ Parsed ${paragraphs.length} paragraphs.`);

// 3️⃣ Create UniversalEncoder for words and sentences
const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyz0123456789",
    mode: "token",
    useTokenizer: true
});

// Word-level embeddings
const wordVectors = paragraphs.map(p => {
    const tokens = p.split(/\s+/).filter(Boolean);
    const tokenVecs = tokens.map(t => encoder.normalize(encoder.encode(t)));
    const avg = tokenVecs[0].map((_, i) =>
        tokenVecs.reduce((s, v) => s + v[i], 0) / tokenVecs.length
    );
    return l2normalize(avg);
});

// Sentence-level embeddings
const sentenceVectors = paragraphs.map(p => {
    const sentences = p.split(/[.?!]\s+/).filter(s => s.length > 3);
    const sentVecs = sentences.map(s => encoder.normalize(encoder.encode(s)));
    const avg = sentVecs[0].map((_, i) =>
        sentVecs.reduce((s, v) => s + v[i], 0) / sentVecs.length
    );
    return l2normalize(avg);
});

// Paragraph-level embeddings
const paragraphVectors = paragraphs.map(p =>
    l2normalize(encoder.normalize(encoder.encode(p)))
);

console.log(`✅ Prepared all input embeddings.`);

// 4️⃣ Helper to train/load a single ELM
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
        elm.trainFromData(vectors, vectors, { reuseWeights: false });
        if (elm.model) {
            fs.writeFileSync(weightsPath, JSON.stringify(elm.model));
            console.log(`💾 Saved weights for ${name}.`);
        }
    }
    return elm;
}

// 5️⃣ Train/load all three encoders
const wordELM = trainOrLoadELM("word_encoder", wordVectors[0].length, wordVectors, 64);
const sentenceELM = trainOrLoadELM("sentence_encoder", sentenceVectors[0].length, sentenceVectors, 64);
const paragraphELM = trainOrLoadELM("paragraph_encoder", paragraphVectors[0].length, paragraphVectors, 128);

// 6️⃣ Compute embeddings for each level
const wordEmb = wordELM.computeHiddenLayer(wordVectors).map(l2normalize);
const sentenceEmb = sentenceELM.computeHiddenLayer(sentenceVectors).map(l2normalize);
const paragraphEmb = paragraphELM.computeHiddenLayer(paragraphVectors).map(l2normalize);

// 7️⃣ Concatenate embeddings
const combinedEmbeddings = wordEmb.map((_, i) =>
    l2normalize([
        ...wordEmb[i],
        ...sentenceEmb[i],
        ...paragraphEmb[i]
    ])
);

console.log(`✅ Combined embeddings.`);

// 8️⃣ Optionally train an Indexer ELM Chain
const hiddenUnitSequence = [256, 128];
let embeddings = combinedEmbeddings;

const indexerELMs = hiddenUnitSequence.map((h, i) =>
    new ELM({
        activation: "relu",
        hiddenUnits: h,
        maxLen: embeddings[0].length,
        categories: [],
        log: { modelName: `IndexerELM_${i}`, verbose: true },
        dropout: 0.02
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
        elm.trainFromData(embeddings, embeddings, { reuseWeights: false });
        if (elm.model) {
            fs.writeFileSync(weightsPath, JSON.stringify(elm.model));
            console.log(`💾 Saved Indexer layer ${i}.`);
        }
    }
    embeddings = elm.computeHiddenLayer(embeddings).map(l2normalize);
});

const indexerChain = new ELMChain(indexerELMs);

// 9️⃣ Save final embeddings
const embeddingRecords: EmbeddingRecord[] = paragraphs.map((p, i) => ({
    embedding: embeddings[i],
    metadata: { text: p }
}));
fs.mkdirSync("./embeddings", { recursive: true });
fs.writeFileSync("./embeddings/combined_embeddings.json", JSON.stringify(embeddingRecords, null, 2));
console.log(`💾 Saved combined embeddings.`);

// 🔍 Example retrieval
function retrieve(query: string, topK = 5) {
    const wordTokens = query.split(/\s+/).filter(Boolean);
    const wordVecs = wordTokens.map(t => encoder.normalize(encoder.encode(t)));
    const avgWord = l2normalize(wordVecs[0].map((_, i) =>
        wordVecs.reduce((s, v) => s + v[i], 0) / wordVecs.length
    ));
    const sentVec = l2normalize(encoder.normalize(encoder.encode(query)));
    const paraVec = sentVec;

    const wordE = wordELM.computeHiddenLayer([avgWord])[0];
    const sentE = sentenceELM.computeHiddenLayer([sentVec])[0];
    const paraE = paragraphELM.computeHiddenLayer([paraVec])[0];
    const combined = l2normalize([...wordE, ...sentE, ...paraE]);

    const finalVec = indexerChain.getEmbedding([combined])[0];

    const scored = embeddingRecords.map(r => ({
        text: r.metadata.text,
        score: r.embedding.reduce((s, v, i) => s + v * finalVec[i], 0)
    }));

    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

// Test retrieval
const results = retrieve("How do you declare a map in Go?");
console.log(`\n🔍 Retrieval results:`);
results.forEach((r, i) =>
    console.log(` ${i + 1}. (${r.score.toFixed(4)}) ${r.text.slice(0, 80)}...`)
);

console.log(`\n✅ Done.`);
